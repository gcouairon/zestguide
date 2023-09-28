import argparse
from cleanfid import fid
from segmentor.eval_miou import SegmentationMetric
from pathlib import Path
from PIL import Image
from torchvision import transforms as TR
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import clip

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, default='', help='folder')
    parser.add_argument("--dataset", type=str, default='coco512', help='dataset')
    parser.add_argument("--metric", type=str, default="", help="which metric to run (by default: all)")
    parser.add_argument("--device", type=str, default='cuda:0', help='device')
    args, unknown = parser.parse_known_args()


    args.metrics = {'fid':True,
                    'clip': True,
                    'miou': True}
    if args.metric:
        args.metrics = {k: (k == args.metric) for k in args.metrics}

    return args


class PairDataset(torch.utils.data.Dataset):
            def __init__(self, folder, prep=None):
                self.folder = Path(folder)
                self.idx = [int(x.name[:-4]) for x in self.folder.joinpath('images').iterdir() if x.name[0]!= '.']
                self.prep = prep or (lambda x: x)
            
            def __len__(self):
                return len(self.idx)
            
            def __getitem__(self, i):
                ix = self.idx[i]
                img = Image.open(self.folder.joinpath(f'images/{ix}.png')).convert('RGB')
                txt = torch.load(self.folder.joinpath(f'logs/{ix}.pt'))['prompt']
                try:
                    clip.tokenize(txt)
                except:
                    txt = txt[:200]
                
                return self.prep(img), txt

class LabelDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.pred_labels = Path(args.folder).joinpath('pred_label')
            self.labels = Path(args.folder).joinpath('labels')
            
            self.idx = [x.name for x in self.pred_labels.iterdir() if not x.name.startswith('.')]

            self.dict_cocostuff = torch.load('data/dict_cocostuff.pt')
            
        
        def __len__(self):
            return len(self.idx)

        def openLbl(self, path, gtlb = False):
            lbl =  Image.open(str(path))
            
            if gtlb:
                lbl = np.array(lbl)
                lbl = np.vectorize(self.dict_cocostuff.get)(lbl)
                lbl = lbl - 1  ### for vit adapter: ignore unlabeled class (set to -1)
                lbl = torch.from_numpy(lbl)
            else:
                lbl = TR.functional.to_tensor(lbl)[0]
            return lbl
        
        def __getitem__(self, i):
            ix = self.idx[i]
            pred_lbl = self.openLbl(self.pred_labels.joinpath(ix))*255
            gt_lbl = self.openLbl(self.labels.joinpath(ix), gtlb=True)
            return pred_lbl, gt_lbl



def evaluate(args):
    metrics = {}

    if args.metrics['fid']:
        # compute fid
        fid_score = fid.compute_fid(str(Path(args.folder).joinpath('images')),
                                dataset_name=args.dataset,
                                mode="clean",
                                dataset_split="custom",
                                device=args.device)
        metrics['fid'] = fid_score
        print('FID is :', fid_score)

    if args.metrics['clip']:
        #compute clipscore
        clip_model, prep = clip.load('ViT-B/32', device=args.device, jit=True)

        ds = PairDataset(folder=args.folder, prep=prep)
        dl = torch.utils.data.DataLoader(ds, num_workers=10, batch_size=32, shuffle=False)

        ps_list = []
        for imgt, txt in tqdm(dl):
            txtt = clip.tokenize(txt)
            imgt_enc = clip_model.encode_image(imgt.to(args.device)).cpu().detach()
            imgt_enc /= imgt_enc.norm(dim=-1, keepdim=True)
            txtt_enc = clip_model.encode_text(txtt.to(args.device)).cpu().detach()
            txtt_enc /= txtt_enc.norm(dim=-1, keepdim=True)
            
            ps_list += (imgt_enc * txtt_enc).sum(-1).tolist()
            
        metrics['clipscore'] = sum(ps_list)/len(ps_list)
        print('CLIP score is: ', metrics['clipscore'])

    if args.metrics['miou']:
        # compute miou
        ds = LabelDataset()
        dl = torch.utils.data.DataLoader(ds, num_workers=10, batch_size=32, shuffle=False)

        seg_metric = SegmentationMetric(172)
        for pred_lbl, gt_lbl in tqdm(dl):
            seg_metric.update(pred_lbl, gt_lbl)

        _, miou = seg_metric.get()
        print('mIoU is: ', miou)
        metrics['mIoU'] = miou
        

    torch.save(metrics, Path(args.folder).joinpath('metrics.pt'))
    


if __name__ == "__main__":
    args = get_args()
    print(f'Starting evaluation of folder {args.folder}')
    evaluate(args)
    