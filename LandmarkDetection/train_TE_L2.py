import argparse
import datetime
import os
from pathlib import Path
import time
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss

from network import UNet, UNet_Pretrained
from data_loader import Cephalometric_IMA
from mylogger import get_mylogger, set_logger_dir
from myTest import Tester
import matplotlib.pyplot as plt
import numpy as np
from metric import total_loss
import torch.nn as nn
from torch.autograd import Variable

def sigmoid_rampup(current, start_es, end_es):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if current < start_es:
        return 0.0
    if current > end_es:
        return 1.0
    else:
        import math
        phase = 1.0 - (current - start_es) / (end_es - start_es)
        return math.exp(-5.0 * phase * phase)

class PGD_TE:
    def __init__(self, num_samples,  momentum, 
                 es, step_size, epsilon, perturb_steps, norm='linf'):
        # initialize soft labels to onthot vectors
        print('number samples: ', num_samples)
        self.num_samples = num_samples
        self.soft_labels = []
        self.momentum = momentum
        self.es = es
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.norm = norm

    def multipleOutputReg(self, Z, Y):
        assert isinstance(Z, (tuple, list)), "x must be either tuple or list"
        assert isinstance(Y, (tuple, list)), "y must be either tuple or list"
        l = ((F.softmax(Z[0].view(Z[0].shape[0], -1), dim=1) - Y[0]) ** 2).mean()
        for i in range(1, len(Z)):
            l +=  ((F.softmax(Z[i].view(Z[i].shape[0], -1), dim=1) - Y[i]) ** 2).mean()
        return l 

    def __call__(self, x_natural,  mask, guassian_mask, offset_y, offset_x, index, epoch, model, optimizer, weight):
        model.eval()
        batch_size = len(x_natural)
        logits = model(x_natural)
        n = len(logits)# should be 3
        
        if len(self.soft_labels) == 0:
            print ("creating the soft labels........ ")
            for i in range(n):
                self.soft_labels.append(torch.zeros(self.num_samples, 
                                                    logits[i].shape[1]*logits[i].shape[2]*logits[i].shape[3], 
                                                    dtype=torch.float).cuda(non_blocking=True))
                
        soft_labels_batch = []
        if epoch >= self.es:           
            for i in range(n):
                prob = F.softmax(logits[i].view(logits[i].shape[0], -1).detach(), dim=1)
                self.soft_labels[i][index] = self.momentum * self.soft_labels[i][index] + (1 - self.momentum) * prob
                soft_labels_batch.append(self.soft_labels[i][index] / self.soft_labels[i][index].sum(1, keepdim=True))

        # generate adversarial example
        if self.norm == 'linf':
            x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-self.epsilon, self.epsilon).cuda()
        else:
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits_adv = model(x_adv)
                if epoch >= self.es:
                    loss = total_loss(logits_adv[0], guassian_mask, logits_adv[1], offset_y, logits_adv[2], offset_x, mask)[0] +  weight * self.multipleOutputReg(logits_adv, soft_labels_batch)
                else:
                    loss = total_loss(logits_adv[0], guassian_mask, logits_adv[1], offset_y, logits_adv[2], offset_x, mask)[0]
            grad = torch.autograd.grad(loss, [x_adv])[0]
            
            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            elif self.norm == 'l2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1,1,1,1)
                scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
                x_adv = x_natural + (x_adv.detach() + 
                                     self.step_size * scaled_grad - 
                                     x_natural).view(x_natural.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(x_natural)                
            else:
                raise "not supported L norm"
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
        
        # compute loss
        model.train()
        optimizer.zero_grad()
        x_adv2 = Variable(torch.clamp(x_adv, -1.0, 1.0), requires_grad=False)     
        # calculate robust loss
        logits = model(x_adv2)
        loss = total_loss(logits[0], guassian_mask, logits[1], offset_y, logits[2], offset_x, mask)[0]
        if epoch >= self.es:
            loss += weight * self.multipleOutputReg(logits, soft_labels_batch)

        return loss

    

if __name__ == "__main__":

    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='TE_L2_',type = str, help="name of the run")
    parser.add_argument("--cuda_id", default=1, type = int, help="cuda id")
    parser.add_argument("--epsilon", default=1.0, type = float, help="training noise")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--max_epochs", default = 500, type = int, help = "default training epochs")
    args = parser.parse_args()
    
    #CUDA_VISIBLE_DEVICES=0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)
 
    # Load yaml config file
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    
    # Create Logger
    logger = get_mylogger()
    logger.info(config)

    # Create runs dir
    args.tag += str(args.epsilon)
    tag = str(datetime.datetime.now()).replace(' ', '_') if args.tag == '' else args.tag
    runs_dir = "./runs/" + tag
    runs_path = Path(runs_dir)
    config['runs_dir'] = runs_dir
    if not runs_path.exists():
        runs_path.mkdir()
    #set_logger_dir(logger, runs_dir)

    dataset = Cephalometric_IMA(config['dataset_pth'], 'Train')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'])
    
    # net = UNet(3, config['num_landmarks'])
    net = UNet_Pretrained(3, config['num_landmarks'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    logger.info(net)

    optimizer = optim.Adam(params=net.parameters(), \
        lr=config['learning_rate'], betas=(0.9,0.999), eps=1e-08, weight_decay=1e-4)
    
    scheduler = StepLR(optimizer, config['decay_step'], gamma=config['decay_gamma'])


    # Tester
    tester = Tester(logger, config, tag=args.tag)
    
    # parameters
 
    #======================
    clip_X_min = -1
    clip_X_max = 1
    noise = args.epsilon
    assert(type(noise) == float)
    noise *= (clip_X_max - clip_X_min)
    norm_type = np.inf
    title = "TE"
    #======================
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()

    #-------------te
    es_start = args.max_epochs * 0.5 # should be 0.5
    es_end = args.max_epochs * 0.7
    noise = args.epsilon  
    pgd_te = PGD_TE( num_samples = len(dataloader.dataset),
                       momentum=0.9,
                       step_size = noise/4,
                       epsilon= noise,
                       perturb_steps=10,
                       norm ='l2',
                       es= es_start)

    for epoch in range(args.max_epochs):
        loss_list = list()
        loss_robust_list = list()
        net.train()
        #----------te
        rampup_rate = sigmoid_rampup(epoch, es_start, es_end)
        weight = rampup_rate * 300
        #----------te
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list, idx in tqdm(dataloader):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()                
            net.zero_grad()           
            loss = pgd_te(img, mask, guassian_mask, offset_y, offset_x, idx, epoch, net, optimizer, weight)          
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

        #-----------------------------------------------------------------------

        loss_train = sum(loss_list) / dataset.__len__()
        loss_train_list.append(loss_train)
        logger.info("Epoch {} Training loss {}".format(epoch, loss_train))       
        
        #validation part       
        MRE, loss_val,loss_logic,  loss_reg = tester.validate(net)
        logger.info("Epoch {} Testing MRE {},  loss {}, logic loss {}, reg loss {}".format(epoch, MRE, loss_val, loss_logic, loss_reg))
        loss_val_list.append(loss_val)
        MRE_list.append(MRE)
        
        
        # save model and plot the trend
        if (epoch + 1) % config['save_seq'] == 0:
            logger.info(runs_dir + "/model_epoch_{}.pth".format(epoch))
            torch.save(net.state_dict(), runs_dir + "/model_epoch_{}.pth".format(epoch))
            # plot the trend
            cols = ['b','g','r','y','k','m','c']
            
            fig,axs = plt.subplots(1,3, figsize=(15,5))

            #ax = fig.add_subplot(111)
            X = list(range(epoch+1))
            axs[0].plot(X, loss_train_list, color=cols[0], label="Training Loss")
            axs[1].plot(X, loss_val_list, color=cols[1], label="Validation Loss")
            axs[2].plot(X, MRE_list, color=cols[2], label="MRE")

            axs[0].set_xlabel("epoch")   
            axs[1].set_xlabel("epoch")
            axs[2].set_xlabel("epoch")

            axs[0].legend()
            axs[1].legend()
            axs[2].legend()           
            fig.savefig(runs_dir +"/training.png")
            #save the last epoch
            config['last_epoch'] = epoch

        # dump yaml
        with open(runs_dir + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    # # Test
    tester.test(net)
