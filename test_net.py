"""Sampling scripts for TiTok on ImageNet.

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

Reference: 
    https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""
"""
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet_rar.py config=configs/training/generator/rar.yaml \
    experiment.output_dir="rar_b" \
    experiment.generator_checkpoint="rar_b.bin" \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072 \
    model.generator.randomize_temperature=1.0 \
    model.generator.guidance_scale=16.0 \
    model.generator.guidance_scale_pow=2.75
    
    

"""
import argparse
import demo_util
import numpy as np
from PIL import Image
from tqdm import tqdm
import os.path as osp
from einops import rearrange
from omegaconf import OmegaConf
import torch.distributed as dist
import os, sys, torch, math, pdb
import tensorflow.compat.v1 as tf
from evaluations.c2i.evaluator import Evaluator
from vfmtok.tokenizer.vq_model import VQ_models
from vfmtok.engine.distributed import init_distributed_mode
from vfmtok.engine.misc import (is_main_process, get_rank, get_world_size, concat_all_gather)

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # setup DDP.
    init_distributed_mode(args)
    config = OmegaConf.load(args.config_file)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    seed = args.global_seed
    rank = get_rank()
    world_size = get_world_size()
    device = rank % torch.cuda.device_count()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if is_main_process():
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.") 

    #* Setup VQ tokenizer
    assert osp.exists(config.model.vq_model.transformer_config_file)
    transformer_config = OmegaConf.load(config.model.vq_model.transformer_config_file)
    tokenizer = VQ_models[config.model.vq_model.tokenizer](
        codebook_size=config.model.vq_model.codebook_size,
        z_channels=config.model.vq_model.z_channels,
        codebook_slots_embed_dim=config.model.vq_model.codebook_slots_embed_dim,
        transformer_config = transformer_config)

    tokenizer.to(device)
    tokenizer.eval()
    tokenizer.freeze()
    checkpoint = torch.load(config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu")

    m1, u1 = tokenizer.load_state_dict(checkpoint["ema"], strict=False)
    del checkpoint
    
    generator = demo_util.get_rar_generator(config, args.gpt_ckpt)
    tokenizer.to(device)
    generator.to(device)

    if args.compile:
        print(f"compiling the model...")
        generator = torch.compile(
            generator,
            mode="reduce-overhead",
            fullgraph=True,
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 
    
    vq_model_name = osp.basename(config.model.vq_model.pretrained_tokenizer_weight).split('.')[0]
    dir_string_name = args.gpt_ckpt.split('/')[2]
    filename = f"{dir_string_name}-size-{args.image_size}-size-{args.image_size_eval}-{vq_model_name}-" \
                  f"guidance-scale-{args.guidance_scale}-guidance-scale-pow-{args.guidance_scale_pow}-seed-{args.global_seed}"

    if is_main_process():
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"Saving .png samples at {osp.join(args.sample_dir, filename)}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    assert args.num_fid_samples % global_batch_size == 0

    if is_main_process():
        print(f"Total number of images that will be sampled: {args.num_fid_samples}")

    samples_needed_this_gpu = int(args.num_fid_samples // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    all_classes = list(range(config.model.generator.condition_num_classes)) * (args.num_fid_samples // config.model.generator.condition_num_classes)
    subset_len = len(all_classes) // world_size
    all_classes = np.array(all_classes[rank * subset_len: (rank+1)*subset_len], dtype=np.int64)
    cur_idx = 0

    gen_samples = []
    for idx in pbar:
        y = torch.from_numpy(all_classes[cur_idx * n: (cur_idx+1)*n]).to(device)
        cur_idx += 1

        samples = demo_util.sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=y.long(),
            randomize_temperature=config.model.generator.randomize_temperature,
            guidance_scale=args.guidance_scale, # guidance_scale=config.model.generator.guidance_scale,
            guidance_scale_pow=args.guidance_scale_pow, #config.model.generator.guidance_scale_pow,
            device=device, return_tensor=True
        )
        samples = concat_all_gather(samples)
        samples = rearrange(samples, 'b c h w -> b h w c')
        samples = torch.clamp(samples, 0.0, 255.0)
        samples = samples.to("cpu", dtype=torch.uint8).numpy()

        '''
        # Save samples to disk as individual .png files
        saveDir = 'images'
        os.makedirs(saveDir, exist_ok=True)
        for i, sample in enumerate(samples):
            index = i * world_size + rank + total
            Image.fromarray(sample).save(f"{saveDir}/{index:06d}.png")
        total += global_batch_size
        '''
        gen_samples.append(samples)


    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if is_main_process():

        gen_samples = np.concatenate(gen_samples, axis=0)[:50_000]
        print(f'generated samples: {gen_samples.shape[0]}')

        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()

        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print("computing/reading reference batch statistics...")
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(gen_samples)
        print("computing/reading sample batch statistics...")
        sample_stats, sample_stats_spatial = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)

        IS = evaluator.compute_inception_score(sample_acts[0])
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

        print("Computing evaluations...")
        print("Inception Score:", IS)
        print("FID:", FID)
        print("sFID:", sFID)
        print("Precision:", prec)
        print("Recall:", recall)

        txt_path = osp.join(args.sample_dir, filename + '.txt')
        print("writing to {}".format(txt_path))
        with open(txt_path, 'w') as f:
            print("Inception Score:", IS, file=f)
            print("FID:", FID, file=f)
            print("sFID:", sFID, file=f)
            print("Precision:", prec, file=f)
            print("Recall:", recall, file=f)

        print("Done.")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--image-size", type=int, choices=[256,336, 384, 512], default=256)
    parser.add_argument('--guidance-scale-pow', type=float, default=2.75)
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--guidance-scale",  type=float, default=16)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=43)

    parser.add_argument("--gpt-ckpt", type=str, default=None)

    # Tokenizer and Generator
    parser.add_argument("--config-file", type=str, default="configs/training/generator/rar.yaml")
    parser.add_argument("--ref-batch", type=str, default='imagenet/VIRTUAL_imagenet256_labeled.npz', help="path to reference batch npz file")
   
    args = parser.parse_args()
    main(args)