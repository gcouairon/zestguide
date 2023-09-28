from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
import torch
from typing import Union, Optional, List, Callable, Tuple, Dict, Any
from diffusers.image_processor import PipelineImageInput
from zestguide_utils import get_idwm, bce_loss, AttentionHook2
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from zestguide_utils import viz

class ZestGuidePipeline(StableDiffusionXLInpaintPipeline):
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            mask_image: PipelineImageInput = None,
            masked_image_latents: torch.FloatTensor = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            strength: float = 0.999,
            num_inference_steps: int = 50,
            denoising_start: Optional[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            tau: float = 0.5, # Gradpaint arg
            lr: float = 0.5, # gradpaint arg
            use_32_blocks=False,
            loss = bce_loss,
            device='cuda:0',
            blur = False,
            use_ddim=False,
            ddim_eta=0.0,
            ediffi_coeff=0,
            segmentation_mask=None,
            idx2words={},
            concat_words_to_prompt=True,
            n_attention_maps=20,
            show=False,
        **kwargs
        ):
            r"""
            Function invoked when calling the pipeline for generation.
    
            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                    instead.
                prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                    used in both text-encoders
                image (`PIL.Image.Image`):
                    `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                    be masked out with `mask_image` and repainted according to `prompt`.
                mask_image (`PIL.Image.Image`):
                    `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                    repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                    to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                    instead of 3, so the expected shape would be `(B, H, W, 1)`.
                height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                    The height in pixels of the generated image. This is set to 1024 by default for the best results.
                    Anything below 512 pixels won't work well for
                    [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                    and checkpoints that are not specifically fine-tuned on low resolutions.
                width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                    The width in pixels of the generated image. This is set to 1024 by default for the best results.
                    Anything below 512 pixels won't work well for
                    [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                    and checkpoints that are not specifically fine-tuned on low resolutions.
                strength (`float`, *optional*, defaults to 0.9999):
                    Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be
                    between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the
                    `strength`. The number of denoising steps depends on the amount of noise initially added. When
                    `strength` is 1, added noise will be maximum and the denoising process will run for the full number of
                    iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked
                    portion of the reference `image`. Note that in the case of `denoising_start` being declared as an
                    integer, the value of `strength` will be ignored.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                denoising_start (`float`, *optional*):
                    When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                    bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                    it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                    strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                    is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refining the Image
                    Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
                denoising_end (`float`, *optional*):
                    When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                    completed before it is intentionally prematurely terminated. As a result, the returned sample will
                    still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                    denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                    final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                    forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                    Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                    `guidance_scale` is defined as `w` of equation 2. of [Imagen
                    Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                    1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                    usually at the expense of lower image quality.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                negative_prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                    `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
                prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                    If not provided, pooled text embeddings will be generated from `prompt` input argument.
                negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                    input argument.
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                    [`schedulers.DDIMScheduler`], will be ignored for others.
                generator (`torch.Generator`, *optional*):
                    One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                    to make generation deterministic.
                latents (`torch.FloatTensor`, *optional*):
                    Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor will ge generated by sampling using the supplied random `generator`.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generate image. Choose between
                    [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                callback (`Callable`, *optional*):
                    A function that will be called every `callback_steps` steps during inference. The function will be
                    called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
                callback_steps (`int`, *optional*, defaults to 1):
                    The frequency at which the `callback` function will be called. If not specified, the callback will be
                    called at every step.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                    `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                    explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                    `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                    `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                    `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    For most cases, `target_size` should be set to the desired height and width of the generated image. If
                    not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                    section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                    To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    To negatively condition the generation process based on a target image resolution. It should be as same
                    as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                aesthetic_score (`float`, *optional*, defaults to 6.0):
                    Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                    Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                    Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                    simulate an aesthetic score of the generated image by influencing the negative text condition.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
    
            Examples:
    
            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
                [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
                `tuple. `tuple. When returning a tuple, the first element is a list with the generated images.
            """
            import torch.nn.functional as F
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            self.lr = lr
            self.tau = tau
            self.ediffi_coeff = ediffi_coeff
            self.loss = loss

            # process idx2words
            segment_weights = torch.ones(len(idx2words), device=device)
            if isinstance(list(idx2words.values())[0], tuple):
                # must be tuple (str, float)
                segment_weights = torch.tensor([x[1] for x in idx2words.values()], device=device)
                segment_weights /= segment_weights.mean()
                idx2words = {k:v[0] for k, v in idx2words.items()}
            if concat_words_to_prompt:
                local_prompt = [x for x in idx2words.values()]
                prompt = prompt + ' ' + ', '.join(local_prompt)
    
            # 1. Check inputs
            self.check_inputs(
                prompt,
                prompt_2,
                height,
                width,
                strength,
                callback_steps,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds,
                negative_prompt_embeds,
            )
    
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
    
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
    
            # 3. Encode input prompt
            with torch.no_grad():
                text_encoder_lora_scale = (
                    cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
                )
        
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    #clip_skip=clip_skip,
                )
    
            # 4. set timesteps
            def denoising_value_valid(dnv):
                return isinstance(denoising_end, float) and 0 < dnv < 1
    
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            
            timesteps = self.scheduler.timesteps
            #timesteps, num_inference_steps = self.get_timesteps(
            #    num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None
            #)
            # check that number of inference steps is not < 1 - as this doesn't make sense
            if num_inference_steps < 1:
                raise ValueError(
                    f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                    f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
                )
            # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
            is_strength_max = strength == 1.0
    
            # 5. Preprocess mask and image
            if image is not None:
                init_image = self.image_processor.preprocess(image, height=height, width=width)
                init_image = init_image.to(dtype=torch.float32)
            else:
                init_image = torch.zeros((1, 3, 1024, 1024))
    
            # 6. Prepare latent variables
            num_channels_latents = self.vae.config.latent_channels
            num_channels_unet = self.unet.config.in_channels
            return_image_latents = num_channels_unet == 4
    
            add_noise = True if denoising_start is None else False
            with torch.no_grad():
                latents_outputs = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                    image=init_image,
                    timestep=latent_timestep,
                    is_strength_max=is_strength_max,
                    add_noise=add_noise,
                    return_noise=True,
                    return_image_latents=return_image_latents,
                )
    
            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs
            self.image_latents = image_latents
        
            # 7. Prepare mask latent variables
            if mask_image is not None:
                mask = self.mask_processor.preprocess(mask_image, height=height, width=width)
        
                if masked_image_latents is not None:
                    masked_image = masked_image_latents
                elif init_image.shape[1] == 4:
                    # if images are in latent space, we can't mask it
                    masked_image = None
                else:
                    masked_image = init_image * (mask < 0.5)
                    
                with torch.no_grad():
                    mask, masked_image_latents = self.prepare_mask_latents(
                        mask,
                        masked_image,
                        batch_size * num_images_per_prompt,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance, # set to False to have a single mask instead of two ?
                    )
            
            # 8.1 Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            height, width = latents.shape[-2:]
            height = height * self.vae_scale_factor
            width = width * self.vae_scale_factor
    
            original_size = original_size or (height, width)
            target_size = target_size or (height, width)
    
            # 10. Prepare added time ids & embeddings
            if negative_original_size is None:
                negative_original_size = original_size
            if negative_target_size is None:
                negative_target_size = target_size
    
            add_text_embeds = pooled_prompt_embeds
            add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
            add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
    
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
    
            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)
    
            # 11. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    
            if (
                denoising_end is not None
                and denoising_start is not None
                and denoising_value_valid(denoising_end)
                and denoising_value_valid(denoising_start)
                and denoising_start >= denoising_end
            ):
                raise ValueError(
                    f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                    + f" {denoising_end} when using type float."
                )
            elif denoising_end is not None and denoising_value_valid(denoising_end):
                discrete_timestep_cutoff = int(
                    round(
                        self.scheduler.config.num_train_timesteps
                        - (denoising_end * self.scheduler.config.num_train_timesteps)
                    )
                )
                num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                timesteps = timesteps[:num_inference_steps]
            # 12. Setup all ZestGuide related stuff
            if len(segmentation_mask.shape) == 2:
                segmentation_mask = segmentation_mask[None]
                
            assert len(segmentation_mask.shape)==3, 'wrong mask shape'
            idWords_inMask = get_idwm(prompt, self.tokenizer, idx2words, segmentation_mask)
            mask_gt = torch.stack([F.interpolate((segmentation_mask == i).float()[None], 32, mode='area')[0, 0]
                           for i, _ in idWords_inMask.items()]).half().to(device)
            from IPython.display import display
            mask_gt = mask_gt.flatten(1)
            self.idWords_inMask = idWords_inMask
        
            # compute ediff masks:
            ediffi_mask_64 = torch.stack([torch.stack([F.one_hot(m.long()*li, num_classes=77) for li in l]).sum(0) 
                            for (i, l), m in zip(idWords_inMask.items(), mask_gt)]).sum(0).half()
            ediffi_mask_64[:, 0] = 0
            
            mask_gt_32 = torch.stack([T.Resize(32, Image.BILINEAR)((segmentation_mask == i).half())[0]
                               for i, _ in idWords_inMask.items()]).to(device).flatten(1)
            
            ediffi_mask_32 = torch.stack([torch.stack([F.one_hot(m.long()*li, num_classes=77) for li in l]).sum(0) 
                            for (i, l), m in zip(idWords_inMask.items(), mask_gt_32)]).sum(0).half()
            ediffi_mask_32[:, 0] = 0
            
            mask_gt_classes = torch.stack([m*i for i, m in zip(idWords_inMask.keys(), mask_gt)]).sum(0).long()
    
            self.logs = logs = {' '.join(w):[m.reshape(1, 32, 32)] for w, m in zip(idx2words.values(), mask_gt)}
            
            attn_hook = AttentionHook2(self.unet, 
                                   ediffi_mask_64=ediffi_mask_64[None], # for attention heads
                                   ediffi_mask_32=ediffi_mask_32[None],
                                   ediffi_coeff=ediffi_coeff,
                                   use_32_blocks=use_32_blocks)
            self.attn_hook = attn_hook
            attn_hook.set_text_embeddings(prompt_embeds[-1:]) # prompt_embeds
        
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    #latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else 
                    latent_model_input = latents
    
                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    scaling_factor = (latent_model_input.pow(2).mean()/latents.pow(2).mean()).sqrt().item()

                    # predict the noise residual
                    
                    # first: do we need uncond ?
                    if do_classifier_free_guidance:
                        with torch.no_grad():
                            added_cond_kwargs = {"text_embeds": add_text_embeds[:1], "time_ids": add_time_ids[:1]}
                            noise_pred_uncond = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds[:1],
                                cross_attention_kwargs=cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]
                    added_cond_kwargs = {"text_embeds": add_text_embeds[-1:], "time_ids": add_time_ids[-1:]} 
                    if i/len(timesteps) < self.tau:
                        #compute gradient
                        latent_model_input.requires_grad_(True)
                        noise_pred_text = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds[-1:],
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                    else:
                        with torch.no_grad():
                            noise_pred_text = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds[-1:],
                                cross_attention_kwargs=cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]
    
                    # perform guidance
                        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if do_classifier_free_guidance:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_text
    
                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
    
                    scheduler_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                    latents = scheduler_output.prev_sample
                    x0_pred = scheduler_output.pred_original_sample
                    
                    #ediffi
                    attn_hook.ediffi_coeff = self.ediffi_coeff * (1 - self.scheduler.alphas_cumprod[t.cpu().int()]).sqrt().detach()
                    #print('ediffi coeff', attn_hook.ediffi_coeff)
                    
                    if show and not(i%5):
                        import torchvision.transforms.functional as F
                        from IPython.display import display
                        self.upcast_vae()
                        with torch.no_grad():
                            im = self.vae.decode(x0_pred.detach().to(next(iter(self.vae.post_quant_conv.parameters())).dtype)/0.13, return_dict=False)[0].detach()
                            im = self.image_processor.postprocess(im, output_type='pil')[0]
                        display(T.Resize(128)(im))
                            
                        self.vae.to(dtype=torch.float16)

                    if i/len(timesteps) < self.tau and lr > 0:
                        # apply zestguide update

                        attention_scores = attn_hook.compute()
                        attention_scores = attention_scores[:, :len(self.tokenizer(prompt)['input_ids'])] # select tokens
                        attention_maps = attention_scores.softmax(1) # can we show the att maps ?

                        #select 10 best maps for each ?
                        
                        #attention_maps = attention_maps.mean((0, 1))
                        attention_maps_for_loss = torch.stack([attention_maps[:, [li for li in l]].sum(1)
                                                               for i, l in idWords_inMask.items()]) # nwords x 32xnmaps
                        # nwords x nmaps x spatial dim
                        # take best maps
                        self.logs['amaps'] = attention_scores.cpu().detach()
                        self.attention_maps_for_loss = attention_maps_for_loss
                        selected_maps = []
                        for j, amlist in enumerate(attention_maps_for_loss):
                            scores = [mp.float().quantile(0.99)*(1-mp.float().quantile(1-0.99))
                                          for mp in amlist]
                            sel_idx = torch.stack(scores).argsort()[-n_attention_maps:]
                            selected_maps.append(amlist[sel_idx].mean(0))
                        attention_maps_for_loss = torch.stack(selected_maps)
                        
                        loss_maps = self.loss(attention_maps_for_loss[:, None], mask_gt)
                        loss = (segment_weights[:, None] * loss_maps).mean()
                        

                        gradient = torch.autograd.grad(outputs=100*loss,
                                                       inputs=latent_model_input)[0]
                        gradient = gradient.detach().requires_grad_(False)

                        gradient_norm = gradient / gradient.max()
                        lr = self.lr * (1 - i/len(timesteps)) # new test
                        
                        latents = latents - lr * gradient_norm / scaling_factor
                        self.unet.zero_grad()
                        # compute miou
                        thres = 0.66
                        pred_miou = attention_maps_for_loss.clip(0, 1) # pred for miou
                        pred_miou = pred_miou / pred_miou.max(dim=-1, keepdim=True).values.add(1e-4) > thres
                        gt_bin = (mask_gt > thres)
                        iou = (pred_miou*gt_bin).sum(1) / (pred_miou + gt_bin).sum(1)
                        miou = iou.mean().cpu().detach()
                        logs.setdefault('miou', [])
                        logs['miou'].append(miou.item())
                        #print('step', i, ', miou', miou.item())
    
                            # maybe display attention_maps here ?
                        if show and not (i % 5):
                            for word, m in zip(idx2words.values(), attention_maps_for_loss.cpu().detach()):
                                logs[' '.join(word)].append(m.reshape(1, 32, 32))
                                print(word)
                                print('m max', m.max())
                                viz(m.reshape((32, 32)), size=128)
            
            
                            print('step', i, ', loss', loss.item())
                    
                    noise_pred_text = noise_pred_text.detach().requires_grad_(False)
                    noise_pred = noise_pred.detach().requires_grad_(False)
                    latent_model_input = latent_model_input.detach().requires_grad_(False)
                    latents = latents.detach().requires_grad_(False)
    
                    if mask_image is not None:
                        if num_channels_unet == 4:
                            init_latents_proper = image_latents
                            if do_classifier_free_guidance:
                                init_mask, _ = mask.chunk(2)
                            else:
                                init_mask = mask
    
                        if i < len(timesteps) - 1:
                            noise_timestep = timesteps[i + 1]
                            inoise = noise if not resample_noise else torch.randn_like(init_latents_proper)
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_proper, inoise, torch.tensor([noise_timestep])
                            )
                            latents = (1 - init_mask) * init_latents_proper + init_mask * latents
    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
            if not output_type == "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                with torch.no_grad():
                    if needs_upcasting:
                        self.upcast_vae()
                        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        
                    # cast back to fp16 if needed
                    if needs_upcasting:
                        self.vae.to(dtype=torch.float16)
            else:
                return StableDiffusionXLPipelineOutput(images=latents)
    
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
    
            image = self.image_processor.postprocess(image, output_type=output_type)
    
            # Offload all models
            self.maybe_free_model_hooks()
    
            if not return_dict:
                return (image,)
    
            return StableDiffusionXLPipelineOutput(images=image)