import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
from IPython.display import display
from collections import OrderedDict
from functools import partial
import torch
from diffusers import StableDiffusionPipeline
# define function to get matrix A.
import collections
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Union, Optional, Callable, List
import numpy as np
from functools import partial
from tqdm import tqdm
from IPython.display import display

def viz(tens, size=None):
    tens = tens.cpu().detach().squeeze()
    if len(tens.shape)==2:
        tens = tens[None]
    if size:
        tens = T.Resize(size)(tens)
    tens /= tens.max()
    display(T.ToPILImage()(tens))

def get_word_inds(text: str, word: str, tokenizer):
    word_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(word)][1:-1]
    text_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
    out = []
    # print('word and text encode is: ', word_encode, text_encode)
    for i,w in enumerate(text_encode):
        if w in word_encode:
            # print('it is in: ', w, word, i)
            out.append(i+1)
    return out

def get_idwm(prompt, tokenizer, placeWords_inMask, spa_mask):
    idWords_inMask = collections.defaultdict(list)
    for i,[k,v] in enumerate(list(placeWords_inMask.items())):
        for word in v.split(' '):
            if word == '<CLS>':
                idWords_inMask[k] += [0]
            p = prompt
            #p = prompts[i] if oneImagePerMask else prompts[0]
            idWords_inMask[k]+=get_word_inds(p, word, tokenizer)
            # print('Idx found for word: ', p, word, idWords_inMask[k])
        if idWords_inMask[k]==[]: ### it's the case if word is '' (unlabeled class of cocostuff)
            idWords_inMask[k]+=[1]
    return idWords_inMask


def mse_loss(att_maps, gts, ls=0.2, **kwargs):
    gts_c = gts.clip(ls, 1-ls)
    att_maps = att_maps.squeeze(1)
    if ls>0:
        disable_grad = ((att_maps - gts).abs()<ls).detach().requires_grad_(False).half()
        att_maps = att_maps * (1 - disable_grad) + disable_grad*gts_c.half()
    
    return (att_maps - gts).pow(2)

def mse_loss2(att_maps, gts, ls=0.2, **kwargs):
    gts_c = gts.clip(ls, 1-ls)
    att_maps = att_maps.squeeze(1)
    if ls > 0:
        disable_grad = ((att_maps - gts).abs()<ls).detach().requires_grad_(False).half()
        att_maps = att_maps * (1 - disable_grad) + disable_grad*gts_c.half()
    
    return (att_maps - gts).pow(4)
    

def bce_loss(att_maps, gts, ls=0.2, eps=1e-3):
    gts_c = gts.clip(ls, 1-ls)
    #ls is for label smoothing
    # test for each map !
    att_maps = att_maps.squeeze(1)
    # avoid pred problem
    att_maps = (att_maps + eps)/(1+2*eps)
    att_maps_norm = att_maps - att_maps.min(dim=-1, keepdim=True).values
    att_maps_norm = att_maps_norm/att_maps_norm.max(dim=-1, keepdim=True).values
    att_maps_norm = (att_maps_norm + eps)/(1+2*eps)
    # disable gradient on very small values
    disable_grad = ((att_maps - gts).abs()<ls).detach().requires_grad_(False).half()
    att_maps2 = att_maps * (1 - disable_grad) + disable_grad*gts_c.half()
    bce_losses = torch.stack([nn.BCELoss(reduction='none')(mp, gt)
                              - nn.BCELoss(reduction='none')(gt, gt)
                              for mp, gt in zip(att_maps2, gts_c)]).mean(0)

    # disable gradient on vsmall values
    disable_grad_norm = ((att_maps_norm - gts).abs()<ls).detach().requires_grad_(False).half()
    att_maps_norm2 = att_maps_norm * (1 - disable_grad_norm) + disable_grad_norm*gts_c.half()
    
    norm_losses = torch.stack([nn.BCELoss(reduction='none')(mp, gt)
                               - nn.BCELoss(reduction='none')(gt, gt)
                               for mp, gt in zip(att_maps_norm2, gts_c)])
    #norm_losses = (norm_losses*(1-disable_grad_norm)).mean(0)
        
    return bce_losses + norm_losses
    
def bce_loss2(att_maps, gts, scaler=5, ls=0.1, eps=1e-3):
    att_maps = (scaler*att_maps.mean(1))
    att_maps_norm = (att_maps/(att_maps.max(dim=-1, keepdim=True).values + 1e-4))
    
    # disable gradient on very small values
    att_maps = att_maps * ((att_maps>0.01) + (gts>0.01)).half().detach().requires_grad_(False)
    bce_losses = torch.stack([nn.BCELoss()(mp.clip(eps, 1-eps), gt.clip(eps, 1-eps)) for mp, gt in zip(att_maps, gts)])
    
    
    weights = torch.ones(gts.shape[0], device = gts.device)
    
    # disable gradient on vsmall values
    att_maps_norm = att_maps_norm * ((att_maps_norm>ls) + (gts>ls)).half().detach().requires_grad_(False)
    
    norm_losses = torch.stack([nn.BCELoss()(mp.clip(eps, 1-eps), gt.clip(eps, 1-eps)) for mp, gt in zip(att_maps_norm, gts)])
    
    weights = weights / weights.mean()
    
    return bce_losses.mean() + norm_losses.mean()
        
        