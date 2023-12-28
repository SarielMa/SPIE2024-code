import argparse
import csv
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
import yaml
import yamlloader
import random

from network import UNet, UNet_Pretrained
from data_loader import Cephalometric
from mylogger import get_mylogger, set_logger_dir
from eval import Evaluater
from utils import to_Image, voting, visualize, make_dir
from attack import FGSMAttack
from torch import optim
from os.path import exists

def cal_dice(Mp, M, reduction='none'):
    #Mp.shape  NxKx128x128
    intersection = (Mp*M).sum(dim=(2,3))
    dice = (2*intersection) / (Mp.sum(dim=(2,3)) + M.sum(dim=(2,3)))
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    return dice

def dice_loss(Mp, M, reduction='none'):
    score=cal_dice(Mp, M, reduction)
    return 1-score

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
#%%
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
def normalize_noise_(noise, norm_type, eps=1e-8):
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        N=noise.view(noise.size(0), -1)
        if norm_type == np.inf or norm_type == 'Linf':
            linf_norm=N.abs().max(dim=1, keepdim=True)[0]
            N *= 1/(linf_norm+eps)
        elif norm_type == 2 or norm_type == 'L2':
            l2_norm=torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            N *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return noise
#%%
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



def L1Loss(pred, gt, mask=None,reduction = "mean"):
    # L1 Loss for offset map
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
        
    if reduction =="mean":
        # sum in this function means 'mean'
        return distence.sum() / mask.sum()
        # return distence.mean()
    else:
        return distence.sum([1,2,3])/mask.sum([1,2,3])
    
def heatmap_dice_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):        
    logic_loss = dice_loss(heatmap, mask, reduction='mean')
    return  logic_loss

    
def reg_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):

    loss_regression_fn = L1Loss
    regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "mean")
    regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "mean")
    return  regression_loss_x + regression_loss_y
    

def total_loss(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask):       
    loss_regression_fn = L1Loss
    logic_loss = dice_loss(heatmap, mask, reduction='mean')
    regression_loss_y = loss_regression_fn(regression_y, offset_y, mask, reduction = "mean")
    regression_loss_x = loss_regression_fn(regression_x, offset_x, mask, reduction = "mean")
    return  logic_loss*0.5 + regression_loss_x + regression_loss_y




#%%
def pgd_attack(net, img, mask, offset_y, offset_x, guassian_mask, noise_norm, norm_type, max_iter, step,
               rand_init=True, rand_init_norm=None,
               clip_X_min=-1, clip_X_max=1, use_optimizer=False, loss_fn=None):
    #-----------------------------------------------------
    if loss_fn is None :
        raise ValueError('loss_fn is unkown')
    #-----------------
    img = img.detach()
    #-----------------
    if rand_init == True:
        init_norm=rand_init_norm
        if rand_init_norm is None:
            init_norm=noise_norm
        noise_init=get_noise_init(norm_type, noise_norm, init_norm, img)
        Xn = img + noise_init
    else:
        Xn = img.clone().detach() # must clone
    #-----------------
    noise_new=(Xn-img).detach()
    #-----------------
    for n in range(0, max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True
        heatmap, regression_y, regression_x = net(Xn)
        loss= loss_fn(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
        grad_n=torch.autograd.grad(loss, Xn)[0]
        grad_n=normalize_grad_(grad_n, norm_type)
        Xnew = Xn.detach() + step*grad_n.detach()
        noise_new = Xnew-img
        #---------------------
        clip_norm_(noise_new, norm_type, noise_norm)
        Xn = torch.clamp(img+noise_new, clip_X_min, clip_X_max)
        Xn=Xn.detach()
    #---------------------------
    return Xn

class Tester(object):
    def __init__(self,logger, testset, noise, norm_type = np.inf):
        self.datapath = "dataset/Cephalometric/"        
        self.nWorkers = 8    
        self.logger = logger
        self.dataset_val = Cephalometric(self.datapath, testset)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.nWorkers)
        self.noise = noise
        self.norm_type = norm_type
        self.clip_X_min = -1
        self.clip_X_max = 1
        self.loss_fn = total_loss

    def validate(self, net, attack):

        dataloader_val = self.dataloader_val
        dataset_val = self.dataset_val
        # initialize a new attack
        evaluater = Evaluater(self.logger, dataset_val.size, dataset_val.original_size)
        # for inference
        Radius = dataset_val.Radius
        # net.eval()
        for img, mask, guassian_mask, offset_y, offset_x, landmark_list in tqdm(dataloader_val):
            img, mask, offset_y, offset_x, guassian_mask = img.cuda(), mask.cuda(), \
                offset_y.cuda(), offset_x.cuda(), guassian_mask.cuda()
            # adversarial attack -----------------------------
            if self.noise > 0:
                net.eval()
                img = self.get_adv(attack, net, img, mask, offset_y, offset_x, guassian_mask)                 
            # get the results---------------------------------
            with torch.no_grad():    
                net.eval()
                heatmap, regression_y, regression_x = net(img)               
                pred_landmark = voting(heatmap, regression_y, regression_x, Radius)   
                evaluater.record(pred_landmark, landmark_list)                
        MRE, _ = evaluater.my_cal_metrics()
        return MRE
    
    def uniform_white_noise(self, net, img, mask, offset_y, offset_x, guassian_mask, noise, norm_type = np.inf):
        # the noise for white test is different from others
        assert norm_type == np.inf
        model = net
        max_iter = 50
        clip_X_min = self.clip_X_min
        clip_X_max = self.clip_X_max
        noise_norm = noise
        loss_fn = self.loss_fn
        with torch.no_grad():
            img_out=img.detach().clone()
            loss_pre = None
            for n in range(0, max_iter):
                imgn = img + noise_norm*(torch.rand_like(img)) # rand_like returns uniform noise in [0,1]
                imgn.clamp_(clip_X_min, clip_X_max)             
                # only record those whose loss is increasing
                with torch.no_grad():
                    heatmap, regression_y, regression_x = model(imgn)
                    loss= loss_fn(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
                if loss_pre is None:
                    loss_pre = loss 
                    img_out = imgn
                if loss > loss_pre:
                    img_out = imgn
                    loss_pre = loss
        return img_out 
 
    def pgd_attack(self, net, img, mask, offset_y, offset_x, guassian_mask, 
                   noise_norm, max_iter, step,
                   norm_type = np.inf, rand_init = True):
        #-----------------------------------------------------
        if self.loss_fn is None :
            raise ValueError('loss_fn is unkown')
        #-----------------
        img = img.detach()
        #-----------------
        if rand_init == True:
            #init_norm=rand_init_norm
            #if rand_init_norm is None:
            init_norm=noise_norm
            noise_init=get_noise_init(norm_type, noise_norm, init_norm, img)
            Xn = img + noise_init
        else:
            Xn = img.clone().detach() # must clone
        #-----------------
        noise_new=(Xn-img).detach()
        #-----------------
        for n in range(0, max_iter):
            Xn = Xn.detach()
            Xn.requires_grad = True
            heatmap, regression_y, regression_x = net(Xn)
            loss= self.loss_fn(heatmap, guassian_mask, regression_y, offset_y, regression_x, offset_x, mask)
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=normalize_grad_(grad_n, norm_type)
            Xnew = Xn.detach() + step*grad_n.detach()
            noise_new = Xnew-img
            #---------------------
            clip_norm_(noise_new, norm_type, noise_norm)
            Xn = torch.clamp(img+noise_new, self.clip_X_min, self.clip_X_max)
            Xn=Xn.detach()
        #---------------------------
        return Xn
    
    def get_adv(self, method, net, img, mask, offset_y, offset_x, guassian_mask):
        noise = self.noise
        if method == 'fgsm':    
            return self.pgd_attack( net, img, mask, offset_y, offset_x, guassian_mask, 
                              noise, max_iter = 1, step = noise, rand_init=False)
        elif method == 'ifgsm10':
            return self.pgd_attack( net, img, mask, offset_y, offset_x, guassian_mask, 
                              noise, max_iter = 10, step =  noise / 5, rand_init=False)
        elif method == 'pgd20':    
            return self.pgd_attack( net, img, mask, offset_y, offset_x, guassian_mask, 
                              noise, max_iter = 20, step =  5*noise / 20, rand_init=True)
        elif method == 'pgd100':
            return self.pgd_attack( net, img, mask, offset_y, offset_x, guassian_mask, 
                              noise, max_iter = 100, step =  5*noise / 100, rand_init=True)
        elif method == "white":
            return self.uniform_white_noise( net, img, mask, offset_y, offset_x, guassian_mask,
                                            noise = 64/255)
        elif method == "natural":
            # no attack is conduct
            return img
        else:
            raise "not implemented"
#%%

def test(noise, paths, attacks):
    loss_fn = total_loss
    
    # Create Logger
    logger = get_mylogger()
    # Load model
       
    print("=======================================================================================")
    rows = []
    for path in paths:
        net = UNet_Pretrained(3, 19).cuda()
        MRE_list =list()
        print ("the model is {}".format(path))
        checkpoints = torch.load(path)
        newCP = dict()
        #adjust the keys(remove the "module.")
        for k in checkpoints.keys():
            newK = ""
            if "module." in k:
                newK = ".".join(k.split(".")[1:])
            else:
                newK = k
            newCP[newK] = checkpoints[k]
        # test
        net.load_state_dict(newCP)
        #net = torch.nn.DataParallel(net)
        print ("this noise level is {}++++++++++++++++++++++++++++++++++++++++++++".format(noise))
        for attack in attacks:
            tester = Tester(logger, testset = args.testset, noise = noise)
            MRE= tester.validate(net, attack = attack)
            print("Attack is {}, result MRE {}+++++++++done+++++++++++++++++++++++".format(attack, MRE))
            MRE_list.append(MRE)
        rows.append([path]+[str(round(i,2)) for i in MRE_list])        
    header = ["MRE"]+[str(i) for i in attacks]
    with open("result_noise"+str(int(noise*255/2))+".csv",'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(rows)

if __name__ == "__main__":
    random.seed(10)

    # Parse command line options
    parser = argparse.ArgumentParser(description="get the threshold from already trained base model")
    parser.add_argument("--tag", default='Grid', help="position of the output dir")
    parser.add_argument("--debug", default='', help="position of the output dir")
    parser.add_argument("--iteration", default='', help="position of the output dir")
    parser.add_argument("--attack", default='', help="position of the output dir")
    parser.add_argument("--config_file", default="config.yaml", help="default configs")
    parser.add_argument("--checkpoint_file", default="", help="default configs")
    parser.add_argument("--output_file", default="", help="default configs")
    parser.add_argument("--train", default="", help="default configs")
    parser.add_argument("--rand", default="", help="default configs")
    parser.add_argument("--epsilon", default=2, type = int, help="default configs")
    parser.add_argument("--testset", default="Test1", type = str)
    parser.add_argument("--cuda_id", default=1,type = int)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id) 
    #file folders================
    folder = "results100"
    models = ["STD.pth",
              "AT2.pth",
              "AT4.pth",
              "AT6.pth",
              "TE2.pth",
              "TE4.pth",
              "TE6.pth",
              "TRADES2.pth",
              "TRADES4.pth",
              "TRADES6.pth"]
    #models = ["STD.pth","AT2.pth"]
    paths = [os.path.join(folder, model) for model in models]
    #========================
    #noises = [0,1,2,3,4]
    #attacks=================================================================
    
    methods = ["white", "fgsm", "ifgsm10", "pgd20", "pgd100"]
    methods = ["natural", "white", "fgsm","ifgsm10", "pgd20", "pgd100"]    
    # check if folders and pts exist=========================================
    for f in paths:
        print ("exist ",f)
        assert( exists(f))
    print ("all files exist, test begins...")
    # test starts
    clip_max = 1
    clip_min = -1
    noise = args.epsilon/255*(clip_max - clip_min)
    test(noise, paths, methods)
        
        
        
 
        
        
        
        
    
    

