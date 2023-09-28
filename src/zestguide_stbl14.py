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
from zestguide_utils import get_idwm, get_word_inds

# def get_word_inds(text: str, word: str, tokenizer):
#     word_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(word)][1:-1]
#     text_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
#     out = []
#     # print('word and text encode is: ', word_encode, text_encode)
#     for i,w in enumerate(text_encode):
#         if w in word_encode:
#             # print('it is in: ', w, word, i)
#             out.append(i+1)
#     return out

# def get_idwm(prompt, tokenizer, placeWords_inMask, spa_mask):
#     idWords_inMask = collections.defaultdict(list)
#     for i,[k,v] in enumerate(list(placeWords_inMask.items())):
#         for word in v.split(' '):
#             if word == '<CLS>':
#                 idWords_inMask[k] += [0]
#             p = prompt
#             #p = prompts[i] if oneImagePerMask else prompts[0]
#             idWords_inMask[k]+=get_word_inds(p, word, tokenizer)
#             # print('Idx found for word: ', p, word, idWords_inMask[k])
#         if idWords_inMask[k]==[]: ### it's the case if word is '' (unlabeled class of cocostuff)
#             idWords_inMask[k]+=[1]
#     return idWords_inMask



   
    


class AttentionHook2:
    def __init__(self, unet, 
                 ediffi_mask_16=0,
                 ediffi_mask_32=0,
                 ediffi_coeff=1,
                 use_32_blocks=False):
        self.__dict__.update(locals())
        self.att_modules = {
        'down_block1':unet.down_blocks[2].attentions[0].transformer_blocks[0].attn2,
         'down_block2':unet.down_blocks[2].attentions[1].transformer_blocks[0].attn2,
        'up_block1':unet.up_blocks[1].attentions[0].transformer_blocks[0].attn2,
        'up_block2':unet.up_blocks[1].attentions[1].transformer_blocks[0].attn2,
           'up_block3': unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2,   
            
}
        if use_32_blocks:
            self.att_modules.update({
                
                'up32_1':unet.up_blocks[2].attentions[0].transformer_blocks[0].attn2,
                'up32_2':unet.up_blocks[2].attentions[1].transformer_blocks[0].attn2,
               'up32_3':unet.up_blocks[2].attentions[2].transformer_blocks[0].attn2})    
        
        self.queries = {mod:None for mod in self.att_modules}
        # remove hooks
        for name, b in self.att_modules.items():
            b._forward_hooks = OrderedDict()
            
            def hook(mod, input, output, name):
                hidden_states = input[0]
                query = mod.to_q(hidden_states)
               
                query = mod.head_to_batch_dim(query)
                
                self.queries[name] = query
            
            
            b.register_forward_hook(partial(hook, name=name))
            
            


        
    def set_text_embeddings(self, text_embeddings):
        self.keys = {key:mod.head_to_batch_dim(mod.to_k(text_embeddings))
                for key, mod in self.att_modules.items()}



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

                
            attention_scores = attention_scores.permute(0, 2, 1)
            
            if attention_scores.shape[-1] == 16*16:
                attention_scores = T.Resize(32)(attention_scores.reshape(8, -1, 16, 16)).flatten(-2)
            elif attention_scores.shape[-1] == 8*8:
                attention_scores = T.Resize(32)(attention_scores.reshape(8, -1, 8, 8)).flatten(-2)

            attention_scores_list.append(attention_scores)

        return (torch.stack(attention_scores_list), 0, 0)
    
            
            
    
class GradientGuidedSynthesis2:
    def __init__(self, 
                 guidance_scale=7.5,
                 num_inference_steps=50,
                 tau=0.5,
                 lr=2,
                 temp=1,
                 use_32_blocks=False,
                 loss = (lambda x:x),
                 device='cuda:0',
                 grad_norm='l0',
                 lr_schedule=False,
                 ddim_eta=0.0,
                 loss_scaler=5,
                 loss_ls=0.1,
                ediffi_coeff=0,
                 clip_init=False,
                 
                 neg_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
                 negative_mode='none',
                **kwargs):
        self.__dict__.update(locals())
        self.neg_prompt = neg_prompt
        self.loss = partial(loss, 
                            scaler=loss_scaler,
                           ls=loss_ls)
        self.pipe = StableDiffusionPipeline.from_pretrained(
       "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16).to(device)
        
    from typing import Union, Optional, Callable, List
    
    def run(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        gen_seed: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        segmentation_mask = None,
        idx2words=None,
        seed=0,
    ):
        generator = gen_seed
        mask = segmentation_mask
        pipe = self.pipe
        torch.set_grad_enabled(False)
        if seed is not None:
            torch.manual_seed(seed)
        # 0. Default height and width to unet
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        idWords_inMask = get_idwm(prompt, pipe.tokenizer, idx2words, mask)

        # 2.1: prepare segmentation stuff
        mask_gt = torch.stack([F.interpolate((mask == i).float()[None], 32, mode='area')[0, 0]
                           for i, _ in idWords_inMask.items()]).half().to(device) # get ground truth 
        
        
        
        mask_gt = mask_gt.flatten(1)
        
        # compute ediff masks:
        ediffi_mask_32 = torch.stack([torch.stack([F.one_hot(m.long()*li, num_classes=77) for li in l]).sum(0) 
                        for (i, l), m in zip(idWords_inMask.items(), mask_gt)]).sum(0)
        ediffi_mask_32[:, 0] = 0
        
        mask_gt_16 = torch.stack([T.Resize(16, Image.BILINEAR)((mask == i).half())[0]
                           for i, _ in idWords_inMask.items()]).to(device).flatten(1)
        
        ediffi_mask_16 = torch.stack([torch.stack([F.one_hot(m.long()*li, num_classes=77) for li in l]).sum(0) 
                        for (i, l), m in zip(idWords_inMask.items(), mask_gt_16)]).sum(0)
        ediffi_mask_16[:, 0] = 0
        
        mask_gt_classes = torch.stack([m*i for i, m in zip(idWords_inMask.keys(), mask_gt)]).sum(0).long()

        logs = {' '.join(w):[m.reshape(1, 32, 32)] for w, m in zip(idx2words.values(), mask_gt)}
        
        # 3. Encode input prompt
        n_tokens = 77#len(pipe.tokenizer(prompt)['input_ids'])

        embeddings = pipe._encode_prompt(prompt, pipe.device, 1, True, self.neg_prompt).detach()[:, :n_tokens]
        null_embeddings, text_embeddings = torch.chunk(embeddings, 2, dim=0)
        
        attn_hook = AttentionHook2(pipe.unet, 
                                   ediffi_mask_32=ediffi_mask_32[None], # for attention heads
                                   ediffi_mask_16=ediffi_mask_16[None],
                                   ediffi_coeff=self.ediffi_coeff,
                                   use_32_blocks=self.use_32_blocks)
        
        text_embs_att = text_embeddings
        negative_embeddings = torch.load('src/coco_class_embeddings.pt')[None]
        # print(self.negative_mode)
        if self.negative_mode == 'concat':
            l = [i for i in range(1, 183) if i not in idx2words.keys()]
            text_embs_att = torch.cat([text_embeddings, negative_embeddings[:, l]], dim=1)
        elif self.negative_mode == 'cls':
            text_embs_att = torch.cat([text_embeddings[:, :1], negative_embeddings[:, 1:]], dim=1)

        attn_hook.set_text_embeddings(text_embs_att)

      
            
        pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipe.unet.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        if self.clip_init:
            latents = latents.clip(-2, 2)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, self.ddim_eta)

        grad_hist = []
        # LOOP ===========================
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance

            if i > int(len(timesteps)*(1 - self.tau)):
                #attn_hook.ediffi_coeff = 0
                extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0)
                torch.set_grad_enabled(False)
                latent_model_input = pipe.scheduler.scale_model_input(latents, t)
                noise_pred_uncond = pipe.unet(latent_model_input, t, encoder_hidden_states=null_embeddings.expand(num_images_per_prompt, *null_embeddings.shape[1:])).sample.detach().requires_grad_(False)
                
                noise_pred_text = pipe.unet(latents, t, encoder_hidden_states=text_embeddings).sample
                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                continue
            
            # here we apply loss
            # extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, self.ddim_eta)
            # attn_hook.ediffi_coeff = self.ediffi_coeff * (1 - pipe.scheduler.alphas_cumprod[t]).sqrt().detach()
            torch.set_grad_enabled(True)
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            latent_model_input.requires_grad_(True)
            noise_pred_text = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.detach()
            attention_scores, queries, keys = attn_hook.compute()
           
            attention_maps = (attention_scores/self.temp).softmax(2).mean((0, 1))
        
            if self.negative_mode == 'cls':
                attention_maps_for_loss = attention_maps[list(idWords_inMask.keys())]
            else:
                attention_maps_for_loss = torch.stack([torch.stack([attention_maps[li] 
                    for li in l]).sum(0) for i, l in idWords_inMask.items()]) # nwords x 32xnmaps    
               
            loss = self.loss(attention_maps_for_loss[:, None], mask_gt)

            with torch.no_grad():
                xt = pipe.scheduler.scale_model_input(latent_model_input, t)
                noise_pred_uncond = pipe.unet(xt, t, 
                                              encoder_hidden_states=null_embeddings.expand(num_images_per_prompt, *null_embeddings.shape[1:])).sample.detach().requires_grad_(False)

            
            scale = self.guidance_scale
            noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

            # #optim
            loss.backward(retain_graph=True)
            grad = latent_model_input.grad
            pipe.unet.zero_grad()
            torch.set_grad_enabled(False)
            gradnorm = grad.abs().max().add(1e-6).detach() if self.grad_norm == 'l0' else grad.abs().sum().detach()
            grad = grad / gradnorm
  
            lr = self.lr
            if self.lr_schedule:
                lr = self.lr * (1 - pipe.scheduler.alphas_cumprod[t]).sqrt().detach()
           
            # print('step', i, ', loss', loss.item())

            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.detach().requires_grad_(False) - lr*grad
                           

            


        # 8. Post-processing
        image = pipe.decode_latents(latents)

        return pipe.numpy_to_pil(image)[0]