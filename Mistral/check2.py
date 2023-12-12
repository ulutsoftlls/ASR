import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import time
# Define the path to the trained model and tokenizer
model_path = "/mnt/ks/Works/t2t/fine-tune-mistral/new_output/5b01669f-f2b2-46c7-b934-2ca2cfb30f6b/epoch_3/step_1758"  # Update with the actual path
tokenizer_path = "/mnt/ks/Works/t2t/fine-tune-mistral/new_output/5b01669f-f2b2-46c7-b934-2ca2cfb30f6b/epoch_3/step_1758"  # Update with the actual path
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
                    device_map="auto",
                    trust_remote_code=True,
                    use_auth_token=True
                )
        # Load the trained model and tokenizer
        #model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=True, trust_remote_code=True)

        # Set the device for inference (e.g., GPU if available)
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device("cuda")

    def generate_text(self, prompt, max_length=300):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        #print(input_ids)
        #print(input_ids['input_ids'])
        with torch.no_grad():
            time_v = time.time()
            generated_tokens = self.model.generate(
                **input_ids, max_length=max_length
            ).to('cuda')
            print("time for output = ", time.time() - time_v)
            generated_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )
        return generated_text

# Example prompt for text generation
# text_generator = TextGenerator()
# prompt = "Жашоонун маңызы эмнеде?"
#
# # Generate text
# generated_text = text_generator.generate_text(prompt)
# print(generated_text)
# prompt = "Дүйнөлүк банк?"
# generated_text = text_generator.generate_text(prompt)
# print(generated_text)
# prompt = 'Манас эпосу?'
# generated_text = text_generator.generate_text(prompt)
# print(generated_text)
#
# prompt = 'Мен киммин?'
# generated_text = text_generator.generate_text(prompt)
# print(generated_text)
