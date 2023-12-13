import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import csv

# Define the path to the trained model and tokenizer
model_path = "/mnt/ks/Works/t2t/fine-tune-mistral/new_output/5b01669f-f2b2-46c7-b934-2ca2cfb30f6b/epoch_3/step_1758"  # Update with the actual path
tokenizer_path = "/mnt/ks/Works/t2t/fine-tune-mistral/new_output/5b01669f-f2b2-46c7-b934-2ca2cfb30f6b/epoch_3/step_1758"  # Update with the actual path
device = "cuda"
class TextGenerator:
    def __init__(self):
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device=device,
            trust_remote_code=True,
            use_auth_token=True
        )
        # Load the trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=True, trust_remote_code=True)

    def generate_text(self, prompts, max_length=100):
        generated_texts = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                time_v = time.time()
                generated_tokens = self.model.generate(
                    input_ids, max_length=max_length
                ).to(device)
                print("time for output = ", time.time() - time_v)
                generated_text = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=True
                )
                generated_texts.append(generated_text)
        return generated_texts

    def save_to_csv(self, prompts, generated_texts, filename='output2.csv'):
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Prompt', 'Generated Text'])
            for prompt, generated_text in zip(prompts, generated_texts):
                writer.writerow([prompt, generated_text])

# Example prompts for text generation
text_generator = TextGenerator()
prompts = ["Жашоонун маңызы эмнеде?", "Кандай?", "Караколдо качан кар жаайт?", "Сен канчадасын?","Мага кандай кеңеш бере аласын?", "Кыргызстан жөнүндө эмне билесин?","Бишкек каякта жайгашкан?", "Сен кимсин?","Эмнени жакшы көрөсүн?", "Кайсыл жерлерде эс алсак болот?","Этиш деген эмне?", "Чыңгыз Айтматов ким?","Садыр Жапаров ким?", "Абанын булганышын кантип азайта алабыз?"," Таттыбүбү Турсунбаева кайсы жылы жана кайсы жерде туулган?"]

# Generate text for each prompt
generated_texts = text_generator.generate_text(prompts)

# Save prompts and generated texts to CSV
text_generator.save_to_csv(prompts, generated_texts)

# Print generated texts
for prompt, generated_text in zip(prompts, generated_texts):
    print(f"Prompt: {prompt}\nGenerated Text: {generated_text}\n")

