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
class KLLoss(nn.Module):
    def __init__(self, ):
        super(KLLoss, self).__init__()
        self.loss = nn.KLDivLoss(size_average = False)

    def forward(self, inputs, target, weight=None, softmax=False):
        batch = inputs[0].size(0)
        n = len(inputs)
        ret = 0
        for i in range(n):
            ret += self.loss(F.log_softmax(inputs[i].view(batch, -1), dim=1), F.softmax(target[i].view(batch, -1), dim=1))
        return ret/n

def trades_loss(model,
                x_natural,
                mask, offset_y, offset_x, guassian_mask,
                optimizer,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_2'):
    # define KL-loss
    clipmax = 1.0
    clipmin = -1.0
    
    step_size = epsilon/5
    
    criterion_kl = KLLoss()
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        epsilon *= (clipmax - clipmin)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(model(x_adv), model(x_natural))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clipmin, clipmax)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta.requires_grad=True
        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            loss = (-1) * criterion_kl(model(x_adv), model(x_natural))
            #loss.backward()
            # renorming gradient
            grad_adv = torch.autograd.grad(loss, x_adv)[0]
            delta.grad=grad_adv.detach()
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(clipmin, clipmax).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        #x_adv = Variable(x_natural + delta, requires_grad=False)
        #x_adv = Variable(torch.clamp(x_adv, clipmin, clipmax), requires_grad=False)
        x_adv = x_natural + delta
        x_adv=x_adv.detach()
    else:
        x_adv = torch.clamp(x_adv, clipmin, clipmax)
    model.train()
    x_adv = torch.clamp(x_adv, clipmin, clipmax) 
    x_adv.requires_grad=True
    #x_adv = Variable(torch.clamp(x_adv, clipmin, clipmax), requires_grad=False)
    optimizer.zero_grad()
    # calculate robust loss  
    heatmap, regression_y, regression_x = model(x_natural)
    loss_natural, _ = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
    #loss_natural = F.cross_entropy(logits, y)
    loss_robust =  criterion_kl(model(x_adv), model(x_natural))/batch_size
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust

    

if __name__ == "__main__":

    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='TRADES_L2_',type = str, help="name of the run")
    parser.add_argument("--cuda_id", default=1, type = int, help="cuda id")
    parser.add_argument("--epsilon", default=5, type = int, help="training noise")
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
    runs_dir = "./runs/" +tag
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
    #assert(type(noise) == float)
    noise *= (clip_X_max - clip_X_min)
    norm_type = np.inf
    max_iter = 10
    title = "TRADES_L2"
    #======================
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()

    for epoch in range(args.max_epochs):
        loss_list = list()
        loss_robust_list = list()
        net.train()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list, idx in tqdm(dataloader):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
                
            net.zero_grad()
            #heatmap, regression_y, regression_x = net(img)
            #lossp, regLossp  = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
            
            loss, loss1, loss2 = trades_loss(net,
                                            img,
                                            mask, offset_y, offset_x, guassian_mask,
                                            optimizer,
                                            epsilon = noise,
                                            perturb_steps = max_iter,
                                            beta = 6.0,
                                            distance = 'l_2')
            
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach())
            loss_robust_list.append(loss2.cpu().detach())
         #--------------------update the margins

        #-----------------------------------------------------------------------

        loss_train = sum(loss_list) / dataset.__len__()
        loss_train_list.append(loss_train)
        logger.info("Epoch {} Training loss {}".format(epoch, loss_train))

        loss_train2 = sum(loss_robust_list) / dataset.__len__()
        logger.info("Epoch {} Training loss {}".format(epoch, loss_train2))        
        
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
