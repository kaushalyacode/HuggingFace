from huggingface_hub import hf_hub_download
import os

HUGGING_FACE_API_KEY = os.environ.get('HUGGING_FACE_API_KEY')

model_id = "google/flan-t5-small"

filenames = [
    "config.json", "flax_model.msgpack", "generation_config.json", "model.safetensors",
    "pytorch_model.bin", "special_tokens_map.json", "spiece.model", "tf_model.h5",
    "tokenizer.json", "tokenizer_config.json"
]

try:
    for filename in filenames:

        downloaded_model_hub = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            token=HUGGING_FACE_API_KEY
        )
        print(f"Downloaded {filename} to {downloaded_model_hub}")
except Exception as e:
    print(f"An error occurred: {e}")


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)


input_text = "translate English to German: How old are you? I want to know what happened around here and need to eat good food."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Translated Text: {translated_text}")

