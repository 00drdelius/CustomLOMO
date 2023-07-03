# for full parameter fine-tuning using LOMO
deepspeed --include localhost:0 src/train_lomo.py config/args_lomo.yaml

# for LoRA + LOMO
#deepspeed --include localhost:0 src/train_lomo_lora.py config/args_lomo_lora.yaml
