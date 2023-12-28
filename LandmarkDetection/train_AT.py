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


def get_noise_init(norm_type, noise_norm, init_norm, X):
    noise_init=2*torch.rand_like(X)-1
    noise_init=noise_init.view(X.size(0),-1)
    if isinstance(init_norm, torch.Tensor):
        init_norm=init_norm.view(X.size(0), -1)
    noise_init=init_norm*noise_init
    noise_init=noise_init.view(X.size())
    clip_norm_(noise_init, norm_type, init_norm)
    clip_norm_(noise_init, norm_type, noise_norm)
    return noise_init

def normalize_grad_(x_grad, norm_type, eps=1e-8):
    #x_grad is modified in place
    #x_grad.size(0) is batch_size
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            x_grad-=x_grad-x_grad.sign()
        elif norm_type == 2 or norm_type == 'L2':
            g=x_grad.view(x_grad.size(0), -1)
            l2_norm=torch.sqrt(torch.sum(g**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            g *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return x_grad


#%%
def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        clip_normA_(noise, norm_type, norm_max)
    else:
        clip_normB_(noise, norm_type, norm_max)
#%%
def clip_normA_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is noise level
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            noise.clamp_(-norm_max, norm_max)
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("other norm clip is not implemented.")
    #-----------
    return noise
#%%
def clip_normB_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max[k] is noise level for every noise[k]
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            #for k in range(noise.size(0)):
            #    noise[k].clamp_(-norm_max[k], norm_max[k])
            N=noise.view(noise.size(0), -1)
            norm_max=norm_max.view(norm_max.size(0), -1)
            N=torch.max(torch.min(N, norm_max), -norm_max)
            N=N.view(noise.size())
            noise-=noise-N
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            norm_max=norm_max.view(norm_max.size(0), 1)
            #print(l2_norm.shape, norm_max.shape)
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                norm_max=norm_max[temp]
                norm_max=norm_max.view(norm_max.size(0), -1)
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("not implemented.")
        #-----------
    return noise

def pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
               noise_norm, norm_type, max_iter, step,
               rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=-1, clip_X_max=1):
    #-------------------------------------------
    #train_mode=net.training# record the mode
    net.eval()#set model to evaluation mode
    #-----------------
    img = img.detach()
    #-----------------
    advc=torch.zeros(img.size(0), dtype=torch.int64, device=img.device)
    #-----------------
    if rand_init_norm is None:
        rand_init_norm = noise_norm
        
    noise_init=get_noise_init(norm_type, noise_norm, rand_init_norm, img)    
    Xn = img + noise_init
    #-----------------
    Xn1=img.detach().clone()
    Xn2=img.detach().clone()
    Ypn_old_e_Y=torch.ones(guassian_mask.shape[0], dtype=torch.bool, device=guassian_mask.device)
    Ypn_old_ne_Y=~Ypn_old_e_Y
    #-----------------
    noise=(Xn-img).detach()
    #-----------------
    for n in range(max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True 
        heatmap, regression_y, regression_x = net(Xn)
        loss, _ = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask) 
        grad_n=torch.autograd.grad(loss, Xn)[0]
        grad_n=normalize_grad_(grad_n, norm_type)
        Xnew = Xn + step*grad_n
        noise = Xnew-img
        #---------------------
        clip_norm_(noise, norm_type, noise_norm)
        Xn = torch.clamp(img+noise, clip_X_min, clip_X_max)

    #---------------------------
    Xn_out = Xn.detach()
    #if train_mode == True and net.training == False:
    net.train()
    #---------------------------
    return Xn_out, advc

    

if __name__ == "__main__":

    #device = torch.device('cuda:1')
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='AT_2',type = str, help="name of the run")
    parser.add_argument("--cuda_id", default=1, type = int, help="cuda id")
    parser.add_argument("--epsilon", default=2, type = int, help="training noise")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--max_epochs", default = 100, type = int, help = "default training epochs")
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
    noise = args.epsilon/255
    assert(type(noise) == float)
    noise *= (clip_X_max - clip_X_min)
    norm_type = np.inf
    max_iter = 7
    step = 5*noise/max_iter
    title = "PGD"
    #======================
    
    loss_train_list = list()
    loss_val_list = list()
    MRE_list = list()

    for epoch in range(args.max_epochs):
        loss_list = list()
        regression_loss_list = list()

        net.train()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list, idx in tqdm(dataloader):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
                
            net.zero_grad()
            #heatmap, regression_y, regression_x = net(img)
            #lossp, regLossp  = total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
            
            imgn, _ = pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, 
                               noise, norm_type, max_iter, step,
                               rand_init_norm=None, rand_init_Xn=None,
                               targeted=False, clip_X_min = clip_X_min, clip_X_max = clip_X_max)
            
            heatmapn, regression_yn, regression_xn = net(imgn)
            lossn, regLossn  = total_loss(heatmapn, guassian_mask, regression_yn, offset_y, regression_xn, offset_x, mask)
            
            loss =  lossn
            
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
         #--------------------update the margins

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
