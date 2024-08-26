MODEL_NAME=stabilityai/stable-video-diffusion-img2vid
TRAIN_DIR=../keyframe_interpolation_data/synthetic_videos_frames
VALIDATION_DIR=eval/val
accelerate launch --mixed_precision="fp16" train_reverse_motion_with_attnflip.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --variant "fp16" \
  --num_frames 14 \
  --train_data_dir=$TRAIN_DIR \
  --validation_data_dir=$VALIDATION_DIR \
  --max_train_samples=100 \
  --train_batch_size=1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs=1000 --checkpointing_steps=2000 \
  --validation_epochs=50 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --double_sampling_rate \
  --output_dir="checkpoints/svd_reverse_motion_with_attnflip" \
  --cache_dir="checkpoints/svd_reverse_motion_with_attnflip_cache" \
  --report_to="tensorboard"
