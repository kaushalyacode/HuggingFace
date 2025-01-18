#pip install huggingface_hub
#pip install transformers
#pip3 install torch torchvision torchaudiof
from huggingface_hub import hf_hub_download
import os

HUGGING_FACE_API_KEY = os.environ.get('HUGGING_FACE_API_KEY')
print(HUGGING_FACE_API_KEY)
model_id = "google/flan-t5-small"
filenames = [
        "config.json","flax_model.msgpack","generation_config.json","model.safetensors","pytorch_model.bin",
    "special_tokens_map.json","spiece.model","tf_model.h5","tokenizer.json","tokenizer_config.json"
]


try:
    for filename in filenames:
        downloaded_model_hub = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            token=HUGGING_FACE_API_KEY
        )
        print(downloaded_model_hub)
except:
    print('done')


from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How old are you I want to know what happned arround here and need to eat good foods?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
