export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="./dataset"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export VALIDATION_PROMPT="A professional photo of a CEO, cinematic, detailed, dramatic lighting."


accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME \
  --resolution=1024 --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=2 \
  --num_train_epochs=10 --checkpointing_steps=4800 \
  --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 --scale_lr \
  --mixed_precision="bf16" \
  --seed=42 \
  --output_dir="sdxl-divjobs-model-lora" \
  --validation_prompt="$VALIDATION_PROMPT" --num_validation_images=16 --report_to="wandb" \
  --rank=64 \
  --dataloader_num_workers=24