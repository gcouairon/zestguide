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
        
class AttentionHook2:
    def __init__(self, unet, 
                 ediffi_mask_32=0,
                 ediffi_mask_64=0,
                 ediffi_coeff=1,
                 use_32_blocks=False):
        self.__dict__.update(locals())
        #self.att_modules = {
            #'mid_block': unet.mid_block.attentions[0].transformer_blocks[0].attn2,
        #'down_block1':unet.down_blocks[2].attentions[0].transformer_blocks[0].attn2,
        # 'down_block2':unet.down_blocks[2].attentions[1].transformer_blocks[0].attn2,
        #'up_block1':unet.up_blocks[1].attentions[0].transformer_blocks[0].attn2,
        #'up_block2':unet.up_blocks[1].attentions[1].transformer_blocks[0].attn2,
           #'up_block3': unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2,   
            
#}
        remove_modules = ['mid']#, 'down_blocks.1', 'up_blocks.1']
        self.att_modules = {n:mod.attn2 for n, mod in unet.named_modules() if hasattr(mod, 'attn2')
                           and not any(rm in n for rm in remove_modules)}
                           
        
        self.queries = {mod:None for mod in self.att_modules}
        # remove hooks
        for name, b in self.att_modules.items():
            b._forward_hooks = OrderedDict()
            
            def hook(mod, input, output, name):
                hidden_states = input[0]
                # batch size should be one
                query = mod.to_q(hidden_states)
                #query = mod.reshape_heads_to_batch_dim(query)
                inner_dim = query.shape[-1]
                head_dim = inner_dim // mod.heads
                batch_size = hidden_states.shape[0]
                query = query.view(1, -1, mod.heads, head_dim).squeeze().transpose(1, 0)

                self.queries[name] = query
                
                # return modified output: no
            
            b.register_forward_hook(partial(hook, name=name))
            
            # change attention (ediff)
                
            def processor(hidden_states, mod=None, hooker=None, **kwargs):
                #new_att_mask = torch.zeros(8, query.shape[1], 77, device=query.device)
                #mask = hooker.ediffi_mask_32 if hidden_states.shape[1] == 1024 else hooker.ediffi_mask_64
                #del kwargs['attention_mask']
                #kwargs['attention_mask'] = hooker.ediffi_coeff * mask[None]
                #print('mask max', hidden_states.shape, mask.shape)
                return mod.processor(mod, hidden_states, 
                                     #attention_mask=hooker.ediffi_coeff * mask[None],
                                     **kwargs)
            
            b.forward = partial(processor, mod=b, hooker=self)


        
    def set_text_embeddings(self, text_embeddings):
        batch_size = text_embeddings.shape[0] # must be one
        self.keys = {key:mod.to_k(text_embeddings)
                for key, mod in self.att_modules.items()}
        self.keys = {key: self.keys[key].view(1, -1, mod.heads, self.keys[key].shape[-1] // mod.heads).squeeze().transpose(1, 0) for key, mod in self.att_modules.items()}

    def compute(self, l2_norm=False, grad_norm=False):
        attention_scores_list = []

        for mod_name, mod in self.att_modules.items():
            key = self.keys[mod_name]
            query = self.queries[mod_name]
            if grad_norm:
                key_norm = key.norm(dim=-1, keepdim=True)
                key = key / (1e-4 + key_norm / key_norm.detach())
                query_norm = query.norm(dim=-1, keepdim=True)
                query = query / (1e-4 + query_norm / query_norm.detach())
            elif l2_norm:
                key_norm = key.norm(dim=-1, keepdim=True)
                key = key / (1e-4 + key_norm / 10)
                query_norm = query.norm(dim=-1, keepdim=True)
                query = query / (1e-4 + query_norm / 10)               
                
                
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                    dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=mod.scale)
            #attention_probs = attention_scores.softmax(dim=-1)
            #nheads x ntok_img x nwords
            # resize to dim 32:


            attention_scores = attention_scores.permute(0, 2, 1)

            sq = int(attention_scores.shape[-1]**.5)
            if sq==64:
                rescale = nn.AvgPool2d(kernel_size=2, stride=2)
                #rescale = T.Resize(64)
                attention_scores = rescale(attention_scores.reshape(mod.heads, -1, sq, sq)).flatten(-2)

            attention_scores_list.append(attention_scores)

        return torch.cat(attention_scores_list)