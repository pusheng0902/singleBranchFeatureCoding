This repository contains the implementation of our single-branch model, channel prunning procedure, and training procedure
of our paper: Compressing multi-scale features with a channel-shrinked single-branch architecture.  
  
Our code is implemented based on Detectron2(https://github.com/facebookresearch/detectron2), with the following changes:  
1. tools/train.py: Implemented the training loop, in which the channel pruning procedure is called.  
2. tools/custom_datasets.py: Implemented a custom dataset loader with no annotations needed.  
3. detectron2/data/dataset_mapper.py: Wrapped the default dataset_mapper with a annotation-free implementation.  
4. detectron2/layers/coder_layers.py: Implemented all NN modules used in the feature coder.  
5. detectron2/utils/prune_utils.py: Implemented all pruning functions.
6. detectron2/modeling/meta_arch/rcnn.py: Added feature reconstruction losses and 
removed unused components (RPN and ROI-heads) from RCNN.  
7. detectron2/modeling/backbone/fpn.py: Implemented feature compression encoder and decoder.
 Added feature coder to the FPN.  
8. configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml and 
configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml: Training hyperparameters  
  
To run our code:
1. First build Detectron2:  
cd detectron2-featureCoding/  
python -m pip install -e detectron2  
  
2. Download pretrained weights of task network from: 
https://drive.google.com/drive/folders/18l1lyASAf6XbIL5ii5teykHzhs6F5xom?usp=drive_link  
  
3. Run training:  
python tools/train.py --config-file configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml  
or  
python tools/train.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml

