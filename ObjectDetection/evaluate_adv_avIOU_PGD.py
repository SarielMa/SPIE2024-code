# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.loss import ComputeLoss
from models.experimental import attempt_load
from torch.cuda import amp
import random
import warnings
warnings.filterwarnings('ignore')

random.seed(10)
#%% adversarial attack section


def parse_opt(model, noise, attackmethod):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/bcc.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / model, help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/testAdv', help='save to project/name')
    parser.add_argument('--name', default='adv', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--this_noise',type = int, default= noise)
    parser.add_argument('--method',type = str, default= attackmethod, help='adversarial attack methods')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    #print_args(FILE.stem, opt)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.device
    return opt

#%%

def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        clip_normA_(noise, norm_type, norm_max)
    else:
        clip_normB_(noise, norm_type, norm_max)

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

scaler = amp.GradScaler(enabled=True)
#%%
class AdversarialAttacks:
    def __init__(self, method, net, noise_norm, norm_type, loss_fn, 
                   rand_init=True, rand_init_norm=None, 
                   clip_X_min=0, clip_X_max=1):
        self.net = net
        self.noise_norm = noise_norm
        self.norm_type = norm_type
        self.loss_fn = loss_fn
        self.rand_init = rand_init
        self.rand_init_norm = rand_init_norm
        self.clip_X_max = clip_X_max
        self.clip_X_min = clip_X_min
        self.method = method   
    def perturb(self, img, gt):
        if self.method == 'ifgsm10':    
            return self.iterative_attack(self.net, img, gt, self.noise_norm, self.norm_type, 10, 5*self.noise_norm/10, self.loss_fn, 
                           rand_init=False, rand_init_norm=None, 
                           clip_X_min=self.clip_X_min, clip_X_max=self.clip_X_max )
            
        elif self.method == 'pgd100':
            return self.iterative_attack(self.net, img, gt, self.noise_norm, self.norm_type, 100, 5*self.noise_norm/100, self.loss_fn, 
                           rand_init=True, rand_init_norm=self.rand_init_norm, 
                           clip_X_min=self.clip_X_min, clip_X_max=self.clip_X_max )
        elif self.method == 'fgsm':    
            return self.iterative_attack(self.net, img, gt, self.noise_norm, self.norm_type, 1, self.noise_norm, self.loss_fn, 
                           rand_init=False, rand_init_norm=None, 
                           clip_X_min=self.clip_X_min, clip_X_max=self.clip_X_max )
        elif self.method == 'pgd20':
            return self.iterative_attack(self.net, img, gt, self.noise_norm, self.norm_type, 20, 5*self.noise_norm/20, self.loss_fn, 
                           rand_init=True, rand_init_norm=self.rand_init_norm, 
                           clip_X_min=self.clip_X_min, clip_X_max=self.clip_X_max )
        elif self.method == "white":
            return self.uniform_white_noise(img, gt)
        else:
            raise "not implemented"
            
        return None
    
    def uniform_white_noise(self, X, Y):
        model = self.net
        max_iter = 50
        clip_X_min = self.clip_X_min
        clip_X_max = self.clip_X_max
        noise_norm = self.noise_norm 
        loss_fn = self.loss_fn
        with torch.no_grad():
            Xout=X.detach().clone()
            loss_pre = None
            for n in range(0, max_iter):
                Xn = X + noise_norm*(torch.rand_like(X)) # rand_like returns uniform noise in [0,1]
                Xn.clamp_(clip_X_min, clip_X_max)             
                # only record those whose loss is increasing
                with torch.no_grad():
                    _, train_out = model(Xn)
                    loss = loss_fn([x.float() for x in train_out], Y)[0]
                if loss_pre is None:
                    loss_pre = loss 
                    Xout = Xn
                if loss > loss_pre:
                    Xout = Xn
                    loss_pre = loss
        return Xout    
    
    def iterative_attack(self, net, img, gt, noise_norm, norm_type, max_iter, step, loss_fn, 
                   rand_init=True, rand_init_norm=None, 
                   clip_X_min=0, clip_X_max=1):
        #-----------------------------------------------------
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
            Xn.requires_grad=True    
            #--------------------------------------------------------
            with torch.enable_grad(): ##this is super strange!!!! without this, Xn cannot get grad though any computing. why ??????
                out, train_out = net(Xn)
                loss = loss_fn([x.float() for x in train_out], gt)[0]
            #---------------------------
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=normalize_grad_(grad_n, norm_type)
            Xnew = Xn.detach() + step*grad_n.detach()
            noise_new = Xnew-img
            #---------------------
            clip_norm_(noise_new, norm_type, noise_norm)
            Xn = torch.clamp(img+noise_new, clip_X_min, clip_X_max)
            #Xn = img + noise_new
            noise_new.data -= noise_new.data-(Xn-img).data
            Xn=Xn.detach()
        #---------------------------
        return Xn


#%% evaluate yolo part

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def get_sample_iou_old(detections, labels):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        matches, including all the ious
    """

    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= 0.5) &(labels[:, 0:1] == detections[:, 5]))  #classes match
    matches = np.array([])
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    detected = len(matches)
    to_detect = labels.size(0)-detected
    if detected>0:
        return np.pad(matches[:,2],[0,to_detect])
    else: 
        return np.zeros(to_detect)
    
def get_sample_iou(detections, labels):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        matches, including all the ious
    """

    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((labels[:, 0:1] == detections[:, 5]))  #classes match
    matches = np.array([])
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    detected = len(matches)
    to_detect = labels.size(0)-detected
    if detected>0:
        return np.pad(matches[:,2],[0,to_detect])
    else: 
        return np.zeros(to_detect)

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=320,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/test',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        this_noise = None,
        method = None
        ):
    
    

    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

    # Data
    data = check_dataset(data)  # check
    
    #device = torch.device('cuda')
    
    
    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader

    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
    #task = 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   stride,
                                   single_cls,
                                   pad=pad,
                                   rect=rect,
                                   workers=workers,
                                   prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    compute_loss = ComputeLoss(model.model)
    
    res = []
    
    #----initial attack
    noise = float(this_noise)/255
    attack = AdversarialAttacks(method, model.model, noise, np.inf, compute_loss, 
                   rand_init=True, rand_init_norm=None, 
                   clip_X_min=0, clip_X_max=1)  
    #----
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # adversarial attack, measured with Linf-norm
        
        if noise>0:
            im = attack.perturb(im, targets)

        # Inference
        out, train_out = model(im)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            res.append( get_sample_iou(predn, labelsn))
            
    return np.concatenate(res).mean()
            






def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    return run(**vars(opt))


def getAUC(ys, xs):
    ret = 0
    for i in range(1,len(xs)):
        h = xs[i]-xs[i-1]
        ret += 0.5*h*(ys[i-1]+ys[i])
    ret /= (xs[-1] - 0)
    return np.sqrt(ys[0]*ret)


def test(noise, models, methods):
    iou_list = []
    for i, model in enumerate(models):
        one_model = [model_names[i]]     
        for method in methods:
            print ("noise ", noise, " method ", method, " model ", model)
            opt = parse_opt(model, noise, method)
            iou = main(opt)
            print ("av iou is ", iou)
            one_model.append(iou)
        #calculate the AUC
        #auc = getAUC(one_method[1:], noises)
        #auc = round(auc, 4)
        #one_method = one_method+[str(auc)]
        iou_list.append(one_model)

   
    all_list = [iou_list]
    
    import csv
    fields1 = ["attacks"]+[i for i in methods]
    with open("avIOU_noise"+str(noise)+".csv",'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields1)
        for line in all_list:
            csvwriter.writerows(line) 
        
if __name__ == "__main__":
    base = "result100"
    models = ["STD_last.pt",
              "AT2_last.pt",
              "AT4_last.pt",
              "AT6_last.pt",
              "TE2_last.pt",
              "TE4_last.pt",
              "TE6_last.pt",
              "TRADES2_last.pt",
              "TRADES4_last.pt",
              "TRADES6_last.pt"]
    models = [
              "TE2_last.pt",
              "TE4_last.pt",
              "TE6_last.pt"]
    #models = ["STD_last.pt"]
    model_names = [name.split("_")[0] for name in models]
    models = [os.path.join(base, path) for path in models]
    for p in models:
        assert(os.path.exists(p))
    print('all the models exist!!!!!!!!!!!!!!!!!!!!!')
    noises = [0,1,2,3,4]
    #noises = [32]
    methods = [ "fgsm", "ifgsm10", "pgd20", "pgd100"]
    #methods = ["white"]
    for noise in noises:
        test(noise, models, methods)
  
    methods = ["white"]
    noises = [32, 64]
    for noise in noises:
        test(noise, models, methods)