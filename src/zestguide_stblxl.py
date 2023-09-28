from zestguide_pipeline import ZestGuidePipeline
import torch
from typing import Union, Optional, Callable, List
import torch.nn as nn
from functools import partial
from zestguide_utils import bce_loss, mse_loss2



class ZestGuide:
    def __init__(self,
                 loss=mse_loss2,
                 ls=0.2,
                device = 'cuda:0', 
                **kwargs):
        
        self.__dict__.update(locals())
        kwargs['loss'] = partial(loss, ls=ls)
        self.kwargs = kwargs
        
        self.pipe =  ZestGuidePipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)

    def run(self,
        prompt: Union[str, List[str]],
        segmentation_mask: torch.FloatTensor,
        idx2words: torch.FloatTensor,
        gen_seed=None):

        return self.pipe(prompt=prompt, 
                         segmentation_mask=segmentation_mask, 
                         idx2words=idx2words, generator = gen_seed,
                        **self.kwargs
                        ).images[0]