import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

peft_model_id = "/mnt/ks/Works/t2t/mistral_instr/results/checkpoint-80000"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
device = "cuda"
# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, trust_remote_code=True
)
tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST] ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
tokenizer.pad_token = tokenizer.unk_token
tokenizer.clean_up_tokenization_spaces = True
tokenizer.add_bos_token = False
tokenizer.padding_side = "right"

model.to(device)

messages = [
    {"role": "user", "content": "Кандай"}
]

print("\n\n*** Prompt:")
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
)
print(tokenizer.decode(input_ids[0]))

print("\n\n*** Generate:")
with torch.autocast("cpu", dtype=torch.bfloat16):
    output = model.generate(
        input_ids=input_ids.to(device),
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=5,
    )

response = tokenizer.decode(
    output["sequences"][0][len(input_ids[0]) :], skip_special_tokens=True
)
print(response)
