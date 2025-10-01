import os 
import re
import json
import argparse 

import torch
from datasets import Dataset,load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from peft import get_peft_model, LoraConfig, TaskType

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def main():
    args = parse_args()

    train_dataset = load_dataset("json", data_files=args.data_path, split="train")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token = args.hug_token,
        trust_remote_code = True,
        torch_dtype = torch.bfloat16,
        )
    
    model.config.use_cache = False
    # model.gradient_checkpointing_enable() # if single-gpu

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        ]
    )
    model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        padding_side="right", 
        use_fast=False
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = SFTConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3, 
        num_train_epochs=3,
        remove_unused_columns=False
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--hug_token', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    main()