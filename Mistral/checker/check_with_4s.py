import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import time
import csv
import tqdm

# Define the path to the trained model and tokenizer
model_path = "/mnt/ks/Works/t2t/fine-tune-mistral/cont_output/2604369c-e818-4756-9e37-db396d9ac427/epoch_2/step_879"  # Update with the actual path
tokenizer_path = "/mnt/ks/Works/t2t/fine-tune-mistral/cont_output/2604369c-e818-4756-9e37-db396d9ac427/epoch_2/step_879"  # Update with the actual path
device = "cuda"
class TextGenerator:
    def __init__(self):
        self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=False,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=self.bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    use_auth_token=True
                )

        # Load the trained model and tokenizer
        #model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=True, trust_remote_code=True)
        device = torch.device(device)
       
        # Set the device for inference (e.g., GPU if available)
        #device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_text(self, prompt,max_length=100):
        device = self.device
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
             time_v = time.time()
        
             generated_tokens = self.model.generate(
                 **input_ids, max_new_tokens=100, do_sample=False, pad_token_id=50256
             ).to(device)
             print("generate time = ", time.time() - time_v)
             generated_text = self.tokenizer.decode(
                 generated_tokens[0], skip_special_tokens=True
             )
             time_c = time.time() - time_v
             print(generated_text)
        return generated_text,time_c      
        
def save_to_csv(prompt, generated_text,filename='output_mistral4s.csv'):
    with open(filename, mode='a', newline='',encoding='utf-8') as file:

        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Prompt', 'Generated Text'])

        generated_text = generated_text[len(prompt)+18:]
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
    save_to_csv(prompt, generated_text)

