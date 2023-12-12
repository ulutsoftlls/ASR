
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Укажите путь к обученному чекпоинту
checkpoint_path = "/mnt/ks/Works/falcon/dataset/kg_llama/checkpoint-4000"

# Загрузите модель и токенизатор
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Введите ваш текст для генерации
prompt = "Ваш текст для генерации..."

# Опции генерации текста
generate_options = {
    "max_length": 50,  # Максимальная длина генерируемого текста
    "num_return_sequences": 1,  # Количество вариантов генерации
    "no_repeat_ngram_size": 2,  # Запрет повторяющихся двойных слов
}

# Подготовьте входные данные для модели
input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)

# Генерация текста
generated_text = model.generate(input_ids, **generate_options)

# Декодирование и печать сгенерированного текста
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print("Сгенерированный текст:")
print(decoded_text)

