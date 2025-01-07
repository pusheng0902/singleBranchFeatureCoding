import base64
import logging
import re
from glob import glob
from pathlib import Path
from typing import Dict, List

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper_noAnno
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.serialize import PicklableWrapper
from PIL import Image
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


def manual_load_data(path, ext):
    img_list = sorted(glob(f"{path}/*.{ext}"))

    datalist = []

    for i, img_addr in tqdm(enumerate(img_list)):
        img_id = Path(img_addr).stem
        img = Image.open(img_addr)
        fW, fH = img.size
        if fW > 1500 and fH > 1500: continue
        ts_name = f'{img_id}.pt'

        d = {
            "file_name": img_addr,
            "height": fH,
            "width": fW,
            "image_id": img_id,
            "annotations": None,
        }

        datalist.append(d)

    return datalist


def bypass_collator(batch):
    return batch


class Detectron2Dataset(Dataset):
    def __init__(self, cfg, img_root):
        super().__init__()

        self.dataset = manual_load_data(img_root, "jpg")
        print('Total number of training images: ', len(self.dataset))

        self.sampler = TrainingSampler(len(self.dataset))

        _dataset = DatasetFromList(self.dataset, copy=False)
        
        mapper = DatasetMapper_noAnno(cfg, True)
        
        self.mapDataset = MapDataset(_dataset, mapper)
        
        self._org_mapper_func = PicklableWrapper(DatasetMapper_noAnno(cfg, True))
        
        self.collate_fn = bypass_collator

    def get_org_mapper_func(self):
        return self._org_mapper_func

    def __getitem__(self, idx):
        return self.mapDataset[idx]

    def __len__(self):
        return len(self.mapDataset)

