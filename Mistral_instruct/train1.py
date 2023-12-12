import json
import pandas as pd
import torch
from torch.cuda import device_count
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
train_dataset = load_dataset('json', data_files='/mnt/ks/Works/t2t/mistral_instr/data_instract/data.jsonl' , split='train')
print(train_dataset)

new_model = "mistralai-Code-Instruct" #set the name of the new model


model_name = "/mnt/ks/Works/t2t/mistral_instr/mistral_model"

tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/ks/Works/t2t/mistral_instr/mistral_model", use_fast=True, trust_remote_code=True
)
tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST] ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
tokenizer.pad_token = tokenizer.unk_token
tokenizer.clean_up_tokenization_spaces = True
tokenizer.add_bos_token = False
tokenizer.padding_side = "right"
tokenizer.pad_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

lora_alpha = 32
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

output_dir = "./results"
num_train_epochs = 25
auto_find_batch_size = True
gradient_accumulation_steps = 1
optim = "paged_adamw_32bit"
save_strategy = "steps"
learning_rate = 1e-5
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
logging_strategy = "steps"
logging_steps = 8000
evaluation_strategy = "epoch"
prediction_loss_only = True
bf16 = True

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    auto_find_batch_size=auto_find_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    logging_strategy=logging_strategy,
    logging_steps=logging_steps,
    evaluation_strategy=evaluation_strategy,
    prediction_loss_only=prediction_loss_only,
    bf16=False,
    fp16=True,
    save_steps=5000
)
max_seq_length = 3072
response_template = "[/INST]"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer, mlm=False
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    data_collator=collator,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
trainer.train()

trainer.save_model("./results/runs/weights/")
