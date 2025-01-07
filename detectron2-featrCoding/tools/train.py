#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.file_io import PathManager

import pickle

from detectron2.utils.prune_utils import *

import numpy as np

from custom_datasets import Detectron2Dataset

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    
    if resume:
        logger.info('Not implemented')
    else:
        # load model
        with PathManager.open(cfg.MODEL.WEIGHTS, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        logger.info("Reading a file from '{}'".format(data["__author__"]))
        for k in data['model'].keys():
            data['model'][k] = torch.Tensor(data['model'][k]).cuda()
        model.load_state_dict(data['model'], strict=False)
        
        # Freeze task model parameters
        for name, param in model.named_parameters():
            # if name.startswith('backbone.Encoder') or name.startswith('backbone.Decoder') or name.startswith('s_'):
            if name.startswith('backbone.Encoder') or name.startswith('backbone.Decoder'):
                print(name, param.requires_grad)
                continue
            param.requires_grad = False
        
        # Freeze BN in NN part1 & 2
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                print('BN freezed!')
        
        # Unfreeze BN in encoder
        for module in model.backbone.Encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                print('BN activated in Enoder!')
        
        # Unfreeze BN in decoder
        for module in model.backbone.Decoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                print('BN activated in Deoder!')
        
        # Sanity check
        for name, param in model.named_parameters():
            print (name, param.requires_grad)
        
        # Set optimizer
        parameters = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad
        ]

        params_dict = dict(model.named_parameters())
        
        optimizer = torch.optim.AdamW(
            (params_dict[n] for n in sorted(parameters)),
            lr=cfg.SOLVER.BASE_LR,
        )
        
        # Set scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(cfg.SOLVER.MAX_ITER))
        
        start_iter = 0
        
    max_iter = cfg.SOLVER.MAX_ITER
    
    
    # build dataloader
    img_root = '/mnt/OpenImages_V7/train'
    dataset = Detectron2Dataset(cfg, img_root)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
            dataset.sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False
    )  # drop_last so the batch always have the same size
    data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=1,
            batch_sampler=batch_sampler,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
    )
    
    # Set Masks
    # Hyperparameters (should be in config file, fix later)
    channel_sparsity = 0.5
    init_channel_ratio = 0.3
    delta_T = 2560
    last_prune = delta_T * 125
    
    cfg_mask, prev_model = init_channel_mask(model.backbone.Decoder, channel_sparsity - init_channel_ratio)
    apply_channel_mask(model.backbone.Decoder, cfg_mask)
    print('apply init. mask | detect channel zero: {}'.format(detect_channel_zero(model.backbone.Decoder)))
    # assert 0
    
    logger.info("Starting training from iteration {}".format(start_iter))
    
    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        # update mask grow prune
        # if iteration == 0: # for debugging
        if iteration > 1 and iteration < last_prune and (iteration + 1) % delta_T == 0:
            channel_ratio = init_channel_ratio * (1 + cos(pi * (iteration) / (last_prune - 1))) / 2 
            layer_ratio_down = [channel_sparsity, channel_sparsity]
            layer_ratio_up = [channel_sparsity - channel_ratio, channel_sparsity - channel_ratio]
            print('layer ratio up:', layer_ratio_up)
            print('layer ratio down:', layer_ratio_down)
            cfg_mask, prev_model = IS_update_channel_mask(model.backbone.Decoder, layer_ratio_up, layer_ratio_down, prev_model)
            apply_channel_mask(model.backbone.Decoder, cfg_mask)
            print('apply updated mask | detect channel zero: {}'.format(detect_channel_zero(model.backbone.Decoder)))
    
        optimizer.zero_grad()
        
        # print(data)
        
        losses, components = model(data)
        
        losses.backward()
        
        optimizer.step()
        scheduler.step()
        
        # maintain channel sparsity
        apply_channel_mask(model.backbone.Decoder, cfg_mask)

        if (iteration + 1) % 20 == 0:
            # print losses
            s = ('iteration=%12s, lr=%f, total=%f, mse_p2=%f, mse_p3=%f, mse_p4=%f, mse_p5=%f, mse_p6=%f') % (
                    iteration, optimizer.param_groups[0]['lr'], losses.item(), components[0],
                    components[1], components[2], components[3], components[4])
            logger.info(s)
        
        
        # Test checkpoint saving:
        if iteration == 0:
            checkpoint = {'iteration': iteration,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(),
                          'cfg_mask': cfg_mask,
                          }
            torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, "iteration_" + str(iteration) + ".pt"))
            print(len(torch.load(os.path.join(cfg.OUTPUT_DIR, "iteration_" + str(iteration) + ".pt"))['cfg_mask'][0]))
            print(len(torch.load(os.path.join(cfg.OUTPUT_DIR, "iteration_" + str(iteration) + ".pt"))['cfg_mask'][1]))
            # assert 0
        
        # Save model:
        if (iteration + 1) > 210000 and (iteration + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0: 
            checkpoint = {'iteration': iteration,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(),
                          'cfg_mask': cfg_mask,
                          }
            torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, "iteration_" + str(iteration) + ".pt"))
            

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    logger.info("Training completed, congrats!")
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
