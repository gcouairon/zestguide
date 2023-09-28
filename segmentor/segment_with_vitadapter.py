# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import mmcv
import torch
from PIL import Image
import cv2
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,wrap_fp16_model)
from mmseg.models import build_segmentor
from PIL import Image
import numpy as np

from pathlib import Path

from mmseg.datasets.pipelines import Compose



def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    # parser.add_argument('--config', type = str,help='test config file path', default = None)
    # parser.add_argument('--checkpoint', type = str, help='checkpoint file', default = None)
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    args = parser.parse_args()
    return args





def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # distributed = True
    # init_dist('pytorch', **cfg.dist_params)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = ADE_CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = ADE_PALETTE
    print(model.CLASSES)
    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    #### Read data to be segmented in image folder
    root = Path(args.work_dir)
    predlbl_path= root.joinpath('pred_label')
    predlbl_path.mkdir(exist_ok=True, parents=True)
    images_path = root.joinpath('images')

    sample_ids = [int(i.name[:-4]) for i in images_path.iterdir() if not i.name.startswith('.')]



    while True:
        processed_files = [int(x.name[:-4]) for x in predlbl_path.iterdir()
                    if not x.name.startswith('.')] # files already processed
        # if len(processed_files.keys()) == len(sample_ids):
        #     break
        if len(sample_ids) == len(processed_files):
            # sample_ids is contained in processed files
            break
        chosen_idx = [i for i in sample_ids if i not in processed_files][0]
        print('len possible idx is: ', len([i for i in sample_ids if i not in processed_files]))
        # choose a file and claim it
        file_name = predlbl_path.joinpath(f'{chosen_idx}.png')
        Image.new('RGB', (1, 1)).save(file_name)

        pipeline = Compose(cfg.data.test.pipeline)
        results = {}
        results['img_info'] = {'filename':f'{chosen_idx}.png'}
        results['seg_fields'] = []
        results['img_prefix'] = images_path
        results['seg_prefix'] = None
        data = pipeline(results)
        data['img'][0] = torch.unsqueeze(data['img'][0], dim=0)
        data['img_metas'][0]._data  = [[data['img_metas'][0]._data]]
        # print('DATA IS: ', [_['ori_shape'] for _ in data['img_metas'][0]])

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # print('size of result: ', np.unique(result[0]),chosen_idx)
            cv2.imwrite(str(predlbl_path.joinpath(f'{chosen_idx}.png')), result[0])


CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood')

PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
           [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
           [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
           [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
           [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
           [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
           [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
           [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
           [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
           [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
           [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
           [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
           [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
           [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
           [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
           [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
           [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
           [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
           [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
           [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
           [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
           [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
           [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
           [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
           [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
           [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
           [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
           [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
           [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
           [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
           [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
           [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
           [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
           [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
           [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
           [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
           [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
           [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
           [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
           [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
           [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
           [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
           [64, 192, 96], [64, 160, 64], [64, 64, 0]]


ADE_CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

ADE_PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
           [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
           [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
           [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
           [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
           [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
           [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
           [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
           [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
           [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
           [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
           [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
           [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
           [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
           [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
           [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
           [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
           [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
           [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
           [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
           [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
           [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
           [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
           [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
           [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
           [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
           [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
           [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
           [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
           [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
           [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
           [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
           [102, 255, 0], [92, 0, 255]]

if __name__ == '__main__':
    args = parse_args()
    main(args)
