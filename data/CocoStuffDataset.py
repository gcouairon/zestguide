import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import json
import collections

class CocoStuffDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, lab_dir, caption_dir, load_size=512):
       
        self.load_size = load_size
        self.data_dir = data_dir 
        self.caption_dir = caption_dir
        self.lab_dir = lab_dir
      
        self.images, self.labels, self.paths = self.list_images()

       
        self.annotations = self.get_annot()

        print('Size of datasets for images/ labels: ', len(self.images), len(self.labels))


    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255 + 1
        label[label == 256] = 0 # unknown class should be zero for correct losses
        # print('label and image: ', label.size(), image.size())
        annot = self.annotations[int(self.images[idx][:-4])]
        return {"image": image, "label": label, "name": self.images[idx], "annotation": annot}

    def get_annot(self):
        with open(f'{self.caption_dir}/captions_val2017.json') as f:
            a = json.load(f)['annotations']
            annot_dict = collections.defaultdict(list)
            for i in a:
                annot_dict[i['image_id']] += [i['caption']]
        return annot_dict

    def list_images(self):
        mode = "val2017" 
        path_img = os.path.join(self.data_dir, mode)
        path_lab = os.path.join(self.lab_dir, mode)
        images = sorted(os.listdir(path_img))
        labels = sorted(os.listdir(path_lab))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    
    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.load_size, self.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label






cocostuff_classes = ['unlabeled',
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'street sign',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'hat',
 'backpack',
 'umbrella',
 'shoe',
 'eye glasses',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'plate',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'mirror',
 'dining table',
 'window',
 'desk',
 'toilet',
 'door',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'blender',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush',
 'hair brush',
 'banner',
 'blanket',
 'branch',
 'bridge',
 'building',
 'bush',
 'cabinet',
 'cage',
 'cardboard',
 'carpet',
 'ceiling',
 'ceiling-tile',
 'cloth',
 'clothes',
 'clouds',
 'counter',
 'cupboard',
 'curtain',
 'desk-stuff',
 'dirt',
 'door-stuff',
 'fence',
 'floor-marble',
 'floor',
 'floor-stone',
 'floor-tile',
 'floor-wood',
 'flower',
 'fog',
 'food',
 'fruit',
 'furniture',
 'grass',
 'gravel',
 'ground',
 'hill',
 'house',
 'leaves',
 'light',
 'mat',
 'metal',
 'mirror-stuff',
 'moss',
 'mountain',
 'mud',
 'napkin',
 'net',
 'paper',
 'pavement',
 'pillow',
 'plant',
 'plastic',
 'platform',
 'playingfield',
 'railing',
 'railroad',
 'river',
 'road',
 'rock',
 'roof',
 'rug',
 'salad',
 'sand',
 'sea',
 'shelf',
 'sky',
 'skyscraper',
 'snow',
 'solid',
 'stairs',
 'stone',
 'straw',
 'structural',
 'table',
 'tent',
 'textile',
 'towel',
 'tree',
 'vegetable',
 'wall-brick',
 'wall-concrete',
 'wall',
 'wall-panel',
 'wall-stone',
 'wall-tile',
 'wall-wood',
 'water',
 'waterdrops',
 'window-blind',
 'window',
 'wood']