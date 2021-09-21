import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

cfg = get_cfg()

# Setting output directory
cfg.OUTPUT_DIR = 'model/'

# Set device cuda/cpu
if torch.cuda.is_available():
    cfg.MODEL.DEVICE = 'cuda'
else:
    cfg.MODEL.DEVICE = 'cpu'
    
print('Setting Device to :', cfg.MODEL.DEVICE)

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


# TRAIING PARAMETERS
cfg.SOLVER.STEPS = []        # do not decay learning rate 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR