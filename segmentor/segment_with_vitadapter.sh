#!/usr/bin/env bash
source /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate vit_adapter
echo "$(which pip)"
echo "$1"

CONFIG="ViT-Adapter/segmentation/configs/coco_stuff164k/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss_stblDif.py"
CHECKPOINT="ViT-Adapter/segmentation/models/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.pth"

python "PATH TO segment_with_vitadapter.py" $CONFIG $CHECKPOINT --work-dir "$1"