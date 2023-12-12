import pandas as pd 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from datasets import Dataset,DatasetDict
import transformers

from peft import LoraConfig


train_df=pd.read_csv("/home/ulan/falcon/dataset/train.csv")
test_df=pd.read_csv("/home/ulan/falcon/dataset/test.csv")


train_dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df),
})


model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","v_proj"]
)
    
from transformers import TrainingArguments

output_dir = "/home/ulan/falcon/dataset/kg_llama"
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
optim = "paged_adamw_32bit"
save_steps = 1000
logging_steps = 1
learning_rate = 1e-5
max_grad_norm = 0.3
max_steps = 50000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)
from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_dict['train'],
    # train_dataset=data['train'],
    peft_config=peft_config,
    dataset_text_field="text",
    # dataset_text_field="prediction",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
        
        
print("Train is started")
trainer.train()
print("Train is finished")




