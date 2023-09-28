# inspired from https://github.com/cloneofsimo/paint-with-words-sd

from PIL import Image, ImageDraw
import numpy as np
import math
import torch
import ast
import gradio as gr
import dotenv
#from paint_with_words import paint_with_words
from zestguide_pipeline import ZestGuidePipeline
import zestguide_utils
import torchvision.transforms as T
from functools import partial


dotenv.load_dotenv()

MAX_NUM_COLORS = 8
pipeline = None

def run_zestguide(color_map_image, init_image, prompt, 
           lr, tau, ls, loss_key, ddim_steps, scale, seed, eta, n_prompt,
           *args):
    # automatically 1024 x 1024
    width = 1024
    height = 1024
    device='cuda:0'
    color_map_image = color_map_image.resize((width, height), Image.Resampling.NEAREST)
    if init_image is not None:
        init_image = init_image.resize((width, height), Image.Resampling.BILINEAR)

    # find idx2words and mask
    idx2words = {}
    n = len(args)
    chunk_size = n // 3
    colors, prompts, strengths = [args[i:i+chunk_size] for i in range(0, n, chunk_size)]
    content_collection = []

    mask = torch.zeros((width, height))
    color_map_tens = torch.from_numpy(np.array(color_map_image)).movedim(-1, 0).int()
    for i, color, prompt, strength in zip(range(chunk_size), colors, prompts, strengths):
        if color != '' and prompt !='':
            idx2words[i+1] = (prompt, float(strength))
            color_arr = torch.tensor([int(x) for x in color[1:-1].split(', ')])
            tidx = (color_map_tens == color_arr[:, None, None]).all(0)
            mask[tidx] = i+1
    global pipeline
    if pipeline is None:
        print('Instantiating Pipeline...')
        pipeline = ZestGuidePipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)

    gen = torch.Generator()
    gen.manual_seed(seed)

    loss_f = zestguide_utils.mse_loss2 if loss_key == 'mse' else zestguide_utils.bce_loss2
    

    imgs = pipeline(prompt=prompt, 
                    init_image=init_image,
                    negative_prompt=n_prompt,
         segmentation_mask=mask,
         idx2words=idx2words,
         generator=gen,
         guidance_scale=scale,
         num_inference_steps=ddim_steps,
         eta=eta,
         lr=lr,
         loss=partial(loss_f, ls=ls),
         ediffi_coeff=0,
         n_attention_maps=20,
         tau=tau).images

    return imgs[0]

def create_canvas(h, w):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

def extract_color_textboxes(color_map_image):
    # Get unique colors in color_map_image

    colors = unique_colors(color_map_image)
    color_masks = [get_color_mask(color, color_map_image) for color in colors]
    # Append white blocks to color_masks to fill up to MAX_NUM_COLORS
    num_missing_masks = MAX_NUM_COLORS - len(color_masks)
    white_mask = Image.new("RGB", color_map_image.size, color=(32, 32, 32))
    color_masks += [white_mask] * num_missing_masks

    default_prompt = ["" for _ in range(len(colors))] + ["" for _ in range(len(colors), MAX_NUM_COLORS)]
    default_strength = ["1" for _ in range(len(colors))] + ["" for _ in range(len(colors), MAX_NUM_COLORS)]
    colors.extend([None] * num_missing_masks)

    visibility = []
    for i in range(MAX_NUM_COLORS):
        visibility.append(gr.Accordion.update(visible=(i<(MAX_NUM_COLORS - num_missing_masks))))

    return (*visibility, *color_masks, *default_prompt, *default_strength, *colors)

def get_color_mask(color, image, threshold=30):
    """
    Returns a color mask for the given color in the given image.
    """
    img_array = np.array(image, dtype=np.uint8)
    color_diff = np.sum((img_array - color) ** 2, axis=-1)
    img_array[color_diff > threshold] = img_array[color_diff > threshold] * 0
    return Image.fromarray(img_array)

def unique_colors(image, threshold=0.01):
    colors = image.getcolors(image.size[0] * image.size[1])
    total_pixels = image.size[0] * image.size[1]
    unique_colors = []
    for count, color in colors:
        if count / total_pixels > threshold and color != (0, 0, 0):
            unique_colors.append(color)
    return unique_colors

def collect_color_content(*args):
    n = len(args)
    chunk_size = n // 3
    colors, prompts, strengths = [args[i:i+chunk_size] for i in range(0, n, chunk_size)]
    content_collection = []
    for color, prompt, strength in zip(colors, prompts, strengths):
        if color is not None and color !='':
            input_str = f'{color}:"{prompt},{strength}"'
            content_collection.append(input_str)
    if len(content_collection) > 0:
        return "{" + ",".join(content_collection) + "}"
    else:
        return ""


if __name__ == '__main__':
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown("## ZestGuide SDXL")
        

        with gr.Row():
            with gr.Column():
                gr.Markdown('### Step 1: Text prompt')    
                prompt = gr.Textbox(label="Text Prompt", show_label=False)
                
                with gr.Row():
                    gr.Markdown('### Step 2: Segmentation Map')
                    gr.Markdown('### (Optional): Initial Image')
                    
                with gr.Row():
                    color_map_image = gr.Image(label='Segmentation map', source='upload', type='pil', tool='color-sketch')
                    init_image = gr.Image(label='Initial image', source='upload', type='pil')

                #color_context = gr.Textbox(label="Color context", value='')
                #run_button = gr.Button(value="Run ZestGuide")            

                extract_color_boxes_button = gr.Button(value="Step 3: Fill in segment descriptions")
                    #generate_color_boxes_button = gr.Button(value="Step 2: Run ZestGuide")
                prompts = []
                strengths = []
                seeds = []
                color_maps = []
                colors = [gr.Textbox(value="", visible=False) for i in range(MAX_NUM_COLORS)]
                accordions = []
                for n in range(MAX_NUM_COLORS):
                    with gr.Accordion(f'Segment {n}', visible=False) as accordion:
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                color_maps.append(gr.Image(interactive=False, type='numpy'))
                            with gr.Column(scale=2, min_width=200):
                                prompts.append(gr.Textbox(label="Prompt", interactive=True))
                                strengths.append(gr.Textbox(label="Strength", interactive=True))
                        accordions.append(accordion)
            with gr.Column():             
                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        lr = gr.Slider(label="Learning Rate", minimum=0, maximum=0.2, value=0.1, step=0.01)
                        tau = gr.Slider(label="Tau", minimum=0, maximum=0.5, value=0.4, step=0.05)
                    with gr.Row():
                        loss_key = gr.Dropdown(label='Loss Function', value='MSE', choices=['MSE', 'BCE'])
                        ls = gr.Slider(label="Label Smoothing", minimum=0, maximum=0.4, value=0.1, step=0.01)
                    with gr.Row():
                        ddim_steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=50, step=5)
                        scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.5)
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=1)
                        eta = gr.Slider(label="eta", minimum=0, maximum=1, value=0., step=0.05)
                        
                    n_prompt = gr.Textbox(label="Negative Prompt", value='')

                run_button = gr.Button(value="Step 4: Run ZestGuide")
                with gr.Row():
                    gr.Markdown("### Results")
                result_gallery = gr.Image(label='Output', show_label=False, interactive=False)
        extract_color_boxes_button.click(fn=extract_color_textboxes, inputs=[color_map_image], outputs=[*accordions, *color_maps, *prompts, *strengths, *colors])
        #generate_color_boxes_button.click(fn=collect_color_content, inputs=[*colors, *prompts, *strengths], outputs=[color_context])

        ips = [color_map_image, init_image, prompt,
              lr, tau, ls, loss_key, ddim_steps, scale, seed, eta, n_prompt,
            *colors, *prompts, *strengths]
        run_button.click(fn=run_zestguide, inputs=ips, outputs=[result_gallery])   
        
    block.launch(server_name='gpu003')