"""Fine-tuning script for Stable Video Diffusion for image2video with support for LoRA."""
import logging
import math
import os
import shutil
from glob import glob
from pathlib import Path
from PIL import Image

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from einops import rearrange
import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
import copy

import diffusers
from diffusers import AutoencoderKLTemporalDecoder
from diffusers import  UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing


from custom_diffusers.pipelines.pipeline_stable_video_diffusion_with_ref_attnmap import StableVideoDiffusionWithRefAttnMapPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (AttentionStore, 
                                         register_temporal_self_attention_control, 
                                         register_temporal_self_attention_flip_control,
)
from utils.parse_args import parse_args
from dataset.stable_video_dataset import StableVideoDataset

logger = get_logger(__name__, log_level="INFO")

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", variant=args.variant
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", variant=args.variant
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True, variant=args.variant
    )
    ref_unet = copy.deepcopy(unet)

    # register customized attn processors
    controller_ref = AttentionStore()
    register_temporal_self_attention_control(ref_unet, controller_ref)

    controller = AttentionStore()
    register_temporal_self_attention_flip_control(unet, controller, controller_ref)

    # freeze parameters of models to save more memory
    ref_unet.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move unet, vae and image_encoder to device and cast to weight_dtype
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    unet_train_params_list = []
    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    for name, para in unet.named_parameters():
        if 'temporal_transformer_blocks.0.attn1.to_v.weight' in name or 'temporal_transformer_blocks.0.attn1.to_out.0.weight' in name:
            unet_train_params_list.append(para)
            para.requires_grad = True
        else:
            para.requires_grad = False
    

    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

     # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if accelerator.is_main_process:
        rec_txt1 = open('frozen_param.txt', 'w')
        rec_txt2 = open('train_param.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet_train_params_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    train_dataset = StableVideoDataset(video_data_dir=args.train_data_dir, 
                                       max_num_videos=args.max_train_samples, 
                                       num_frames=args.num_frames,
                                       is_reverse_video=True,
                                       double_sampling_rate=args.double_sampling_rate)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        conditions = torch.stack([example["conditions"] for example in examples])
        conditions =conditions.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "conditions": conditions}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Validation data
    if args.validation_data_dir is not None:
        validation_image_paths = sorted(glob(os.path.join(args.validation_data_dir, '*.png')))
        num_validation_images = min(args.num_validation_images, len(validation_image_paths))
        validation_image_paths = validation_image_paths[:num_validation_images]
        validation_images = [Image.open(image_path).convert('RGB').resize((1024, 576)) for image_path in validation_image_paths]

        
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("image2video-reverse-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    # default motion param setting
    def _get_add_time_ids(
        dtype,
        batch_size,
        fps=6,
        motion_bucket_id=127,
        noise_aug_strength=0.02,  
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = unet.module.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features
        assert (expected_add_embed_dim == passed_add_embed_dim)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def compute_image_embeddings(image):
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        # Normalize the image with for CLIP input
        image = feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values
        
        image = image.to(accelerator.device).to(dtype=weight_dtype)
        image_embeddings = image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)
        return image_embeddings

    noise_aug_strength = 0.02
    fps=7         
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get the image embedding for conditioning
                encoder_hidden_states = compute_image_embeddings(batch["conditions"])
                encoder_hidden_states_ref = compute_image_embeddings(batch["pixel_values"][:, -1])
                
                batch["conditions"] = batch["conditions"].to(accelerator.device).to(dtype=weight_dtype)
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device).to(dtype=weight_dtype)
        
                # Get the image latent for input condtioning
                noise =  torch.randn_like(batch["conditions"])
                conditions = batch["conditions"] + noise_aug_strength * noise
                conditions_latent = vae.encode(conditions).latent_dist.mode()
                conditions_latent = conditions_latent.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)

                conditions_ref = batch["pixel_values"][:, -1] + noise_aug_strength * noise
                conditions_latent_ref = vae.encode(conditions_ref).latent_dist.mode()
                conditions_latent_ref = conditions_latent_ref.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)

                # Convert frames to latent space
                pixel_values = rearrange(batch["pixel_values"], "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = rearrange(latents, "(b f) c h w -> b f c h w", f=args.num_frames)
                latents_ref= torch.flip(latents, dims=(1,))

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], latents.shape[2], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = latents + noise * sigmas
                noisy_latents_inp = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                noisy_latents_inp = torch.cat([noisy_latents_inp, conditions_latent], dim=2)

                noisy_latents_ref = latents_ref + torch.flip(noise, dims=(1,)) * sigmas
                noisy_latents_ref_inp = noisy_latents_ref / ((sigmas**2 + 1) ** 0.5)
                noisy_latents_ref_inp = torch.cat([noisy_latents_ref_inp, conditions_latent_ref], dim=2)

                # Get the target for loss depending on the prediction type
                target = latents
                # Predict the noise residual and compute loss
                added_time_ids = _get_add_time_ids(encoder_hidden_states.dtype, bsz).to(accelerator.device)
                ref_model_pred = ref_unet(noisy_latents_ref_inp.to(weight_dtype), timesteps.to(weight_dtype),
                                encoder_hidden_states=encoder_hidden_states_ref, 
                                added_time_ids=added_time_ids, 
                                return_dict=False)[0]
                model_pred = unet(noisy_latents_inp, timesteps,
                                encoder_hidden_states=encoder_hidden_states, 
                                added_time_ids=added_time_ids, 
                                return_dict=False)[0] # v-prediction
                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                 # MSE loss
                loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() -
                        target.float()) ** 2).reshape(target.shape[0], -1),
                        dim=1,
                )
                loss = loss.mean()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_train_params_list
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_data_dir is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_data_dir}."
                )
                # create pipeline
                pipeline = StableVideoDiffusionWithRefAttnMapPipeline.from_pretrained(
                    args.pretrained_model_name_or_path, 
                    scheduler=noise_scheduler,
                    unet=unwrap_model(unet),
                    variant=args.variant,
                    torch_dtype=weight_dtype, 
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                videos = []
                with torch.cuda.amp.autocast():
                    for val_idx in range(num_validation_images):
                        val_img = validation_images[val_idx]
                        videos.append(
                            pipeline(ref_unet=ref_unet, image=val_img, ref_image=val_img, num_inference_steps=50, generator=generator, output_type='pt').frames[0]
                        )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        videos = torch.stack(videos)
                        tracker.writer.add_video("validation", videos, epoch, fps=fps)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        pipeline = StableVideoDiffusionWithRefAttnMapPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    scheduler=noise_scheduler,
                    unet=unwrapped_unet,
                    variant=args.variant,
                )
        pipeline.save_pretrained(args.output_dir)    
        # Final inference
        # Load previous pipeline
        if args.validation_data_dir is not None:
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            # run inference
            generator = torch.Generator(device=accelerator.device)
            if args.seed is not None:
                generator = generator.manual_seed(args.seed)
            videos = []
            with torch.cuda.amp.autocast():
                for val_idx in range(num_validation_images):
                    val_img = validation_images[val_idx]
                    videos.append(
                        pipeline(ref_unet=ref_unet, image=val_img, ref_image=val_img, num_inference_steps=50, generator=generator, output_type='pt').frames[0]
                    )


            for tracker in accelerator.trackers:
                if len(videos) != 0:
                    if tracker.name == "tensorboard":
                        videos = torch.stack(videos)
                        tracker.writer.add_video("validation", videos, epoch, fps=fps)
                    
    accelerator.end_training()


if __name__ == "__main__":
    main()
