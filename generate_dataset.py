# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from utils.io import load, dump, cmd_args_to_dict
from utils.load_config import instantiate
from utils.io import load
import subprocess

from data.CocoStuffDataset import CocoStuffDataset, cocostuff_classes


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default='exp_configs/zestguide.yaml', help='config file')
    parser.add_argument("--output", type=str, default='generated/default/', help='where to store images')
    parser.add_argument("--eval_setting", type=str, default='eval_few', help='which eval setting to use; choose between eval_all, eval_filtered, eval_few')
    parser.add_argument("--submitit", action='store_true', help='whether to use submitit')
    parser.add_argument("--dataset", type=str, default='coco', help='which dataset to eval')
    parser.add_argument("--nsamples", type=int, default=5000, help='n samples limit')

    parser.add_argument("--ngpus", type=int, default=16, help='# GPUs')
    parser.add_argument("--seed", type=int, default=0,help='seed')
    
    
    
    args, unknown = parser.parse_known_args()
    cmd_args = cmd_args_to_dict(unknown)
    return args, cmd_args


def main(args):
    
    list_classes = cocostuff_classes 
    dataset = CocoStuffDataset(data_dir=args.data_dir, lab_dir = args.lab_dir, caption_dir=args.caption_dir, load_size = args.size)
  
    generator = args.model()
    
    
    n_imgs = args.nsamples
    dataset_ids = list(range(0, len(dataset), len(dataset)//n_imgs))[:n_imgs] 


    gen_seed = torch.Generator()
    gen_seed.manual_seed(args.seed)  
   
     
    while True:
        processed_files = {int(x.name[:-3]):True for x in args.logs_folder.iterdir()
                    if not x.name.startswith('.')} # files already processed
        if len(processed_files.keys()) == len(dataset_ids):
            break
        chosen_idx = [i for i in dataset_ids if i not in processed_files][0]
        # choose a file and claim it
        file_name = args.images_folder.joinpath(f'{chosen_idx}.png')
        torch.save({}, args.logs_folder.joinpath(f'{chosen_idx}.pt'))
        
        # load image and data
        batch = dataset[chosen_idx]
        gt_im = batch['image']

        # generate spatial mask stuff
        spa_mask = batch['label']
        idx_lbl, count_lbl = torch.unique(spa_mask, return_counts=True)
        
        # filter spatial mask depending on the eval setting (see zestguide paper, Table 1)
        if not args.eval_setting=='eval_all':
            # remove unlabeled class and small objects:
            idx_lbl = idx_lbl[count_lbl > 5 / 100 * spa_mask.size(-2) * spa_mask.size(1)]
            idx_lbl = idx_lbl[idx_lbl != 0]
            
        if idx_lbl.size(0) == 0:
            continue
        
        if args.eval_setting=='eval_few':
            # select between 1 and 3 segments in the mask
            min_segm = 1
            max_segm = 3
            if min_segm == idx_lbl.size(0):
                K = min_segm
            else:
                K = torch.randint(min_segm, min(max_segm+1, idx_lbl.size(0)+1), (spa_mask.size(0),))
            rnd_select_idx = torch.randperm(idx_lbl.size(0))[:K]
            selected_cls = idx_lbl[rnd_select_idx]
        else:
            selected_cls = idx_lbl
        
        # idx2words = {int(i.item()): list_classes[int(i.item())].replace('-stuff', '').replace('-', ' ').split(" ") for i in selected_cls}
        idx2words = {int(i.item()): list_classes[int(i.item())].replace('-stuff', '').replace('-', ' ') for i in selected_cls}
        selected_mask = torch.stack([i * (spa_mask == i).float() for i in idx2words.keys()]).sum(0)

        # create conditioning prompt
        caption = batch['annotation'][0]
        # local_prompt = [' '.join(x) for x in idx2words.values()]
        local_prompt = [x for x in idx2words.values() if x not in caption]
        prompt = caption + ' ' + ', '.join(local_prompt)
        # print('PROMPT IS: ', prompt, idx2words)
       
        # generate image
        img = generator.run(prompt, segmentation_mask=selected_mask,
                                  idx2words=idx2words,
                                 gen_seed=gen_seed)

        # save generated image
        img.save(file_name)

        # save selected mask
        selected_mask = selected_mask.numpy().astype(np.uint8)
        img = Image.fromarray(selected_mask[0])
        img.save(args.labels_folder.joinpath(f'{chosen_idx}.png'))      
       

        torch.save({'original_caption': caption, 'idx2words':idx2words, 'prompt':prompt}, args.logs_folder.joinpath(f'{chosen_idx}.pt'))




        T.ToPILImage()((gt_im+1)/2).save(args.gtimages_folder.joinpath(f'{chosen_idx}.png'))

    # compute predicted lbl w/ ViT-Adapter
    bashCommand = f"bash segmentor/segment_with_vitadapter.sh {str(args.output)}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)

    # evaluate 
    eval_file = args.folder.joinpath('metrics.pt')
    if not eval_file.exists():
        print('Creating file')
        torch.save({}, eval_file)
        bashCommand = f"python evaluate.py {str(args.output)} --dataset {args.dataset+args.size}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)



if __name__ == '__main__':
    args, cmd_args = get_parser()
    file_args = load(args.config) #returns dict
    args.__dict__.update(file_args)
    args.__dict__.update(cmd_args)
    
    Path(args.output).mkdir(exist_ok=True, parents=True)
    
    dump(args.__dict__, Path(args.output).joinpath('hparams.yaml').__str__())
    
    args.output = Path(args.output)
    args.images_folder = args.output.joinpath('images')
    args.gtimages_folder = args.output.joinpath('gt_images')
    args.logs_folder = args.output.joinpath('logs')
    args.labels_folder = args.output.joinpath('labels')
    
    
    args.images_folder.mkdir(exist_ok=True, parents=True)
    args.gtimages_folder.mkdir(exist_ok=True, parents=True)
    args.logs_folder.mkdir(exist_ok=True, parents=True)
    args.labels_folder.mkdir(exist_ok=True, parents=True)
                     
    args = instantiate(args)
    
    
    if args.submitit:
        from utils.submit import submit
        submit(main, 
           args, 
           output=args.output, 
           ngpus=args.ngpus)
    else:
        # quick test
        main(args)