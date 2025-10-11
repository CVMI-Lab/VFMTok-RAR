"""Training script for RAR.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
import numpy as np
import os.path as osp
import math, torch, pdb
from pathlib import Path
import os, sys, argparse

from omegaconf import OmegaConf
from accelerate import Accelerator

from utils.logger import setup_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from vfmtok.tokenizer.vq_model import VQ_models
from torch.utils.data.distributed import DistributedSampler
from vfmtok.data.imagenet_lmdb import ImageNetLmdbDataset as ImageNetDataset
from vfmtok.engine.misc import is_main_process, all_reduce_mean, concat_all_gather,get_world_size, get_rank

from utils.train_utils import (get_config, create_model_and_loss_module, create_pretrained_tokenizer,
                        create_optimizer, create_lr_scheduler, create_dataloader, auto_resume, 
                        save_checkpoint, train_one_epoch_generator, batch_data_collate)


def main(args):

    workspace = os.environ.get('WORKSPACE', '')

    config = OmegaConf.load(args.config_file)
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = osp.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    # rank = get_rank()
    logger = setup_logger(name="RAR", log_level="INFO", output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("RAR")
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)


    #* Setup VQ tokenizer
    assert osp.exists(config.model.vq_model.transformer_config_file)
    transformer_config = OmegaConf.load(config.model.vq_model.transformer_config_file)
    tokenizer = VQ_models[config.model.vq_model.tokenizer](
        codebook_size=config.model.vq_model.codebook_size,
        z_channels=config.model.vq_model.z_channels,
        codebook_slots_embed_dim=config.model.vq_model.codebook_slots_embed_dim,
        transformer_config = transformer_config)

    tokenizer.to(accelerator.device)
    tokenizer.eval()
    tokenizer.freeze()
    checkpoint = torch.load(config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu")

    m1, u1 = tokenizer.load_state_dict(checkpoint["ema"], strict=False)
    del checkpoint

    #* Setup dataset/dataloader:
    dataset = ImageNetDataset(args.anno_file, args.image_size, True)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        seed=config.training.seed,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        sampler=sampler, collate_fn=batch_data_collate,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    train_dataloader.num_batches = num_batches = math.ceil(config.experiment.max_train_examples / total_batch_size_without_accum)

    logger.info(f"Dataset contains {len(train_dataloader):,} images ({args.anno_file}).")

    #* Setup RAR model.
    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="rar")

    optimizer, _ = create_optimizer(config, logger, model, loss_module,
                                    need_discrminator=False)

    lr_scheduler, _ = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer=None)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    
    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=True)

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        
        train_dataloader.sampler.set_epoch(current_epoch)
        global_step = train_one_epoch_generator(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer,
                            lr_scheduler,
                            train_dataloader,
                            tokenizer,
                            global_step,
                            model_type="rar")
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint

    accelerator.end_training()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Tokenizer
    parser.add_argument("--config-file", type=str, default="configs/training/generator/rar.yaml")
    
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 448, 512], default=256)
    parser.add_argument("--anno-file", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    main(args)