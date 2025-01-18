# Prerequisites

Ensure you have Python 3.7 or later installed. Install the required libraries before running the script.

Used Version : Python 3.13.1    
IDE : PyCharm

## 1. Install Dependencies
Run the following commands to install the necessary libraries:
```bash
pip install huggingface_hub
pip install transformers
pip3 install torch torchvision torchaudio
```
### huggingface_hub
This allow application to interact with Hugging face hub.   
In this exmple:     
    ```hf_hub_download``` has used for download files from the repositories stored on the Hub. 

### transformers

This installs the Transformers library, a powerful toolset for using pre-trained models and building state-of-the-art deep learning pipelines for tasks like NLP, computer vision, and more.

### torch torchvision torchaudio

This installs PyTorch along with its vision (torchvision) and audio (torchaudio) libraries, which provide utilities for deep learning and multimodal applications.

## 2. Obtain Your API Key:

2.1 Log in to your Hugging Face account.    
2.2 Navigate to Settings > Access Tokens.   
2.3 Click New Token, name it, select the role (e.g., Read), and generate.   
2.4 Copy the generated token and store it securely.

In this case we have stored variable in machines environment variable.```HUGGING_FACE_API_KEY```

## Explained
```
from huggingface_hub import hf_hub_download
import os

# Retrieve the Hugging Face API key from environment variables
HUGGING_FACE_API_KEY = os.environ.get('HUGGING_FACE_API_KEY')

# Define the model identifier
model_id = "google/flan-t5-small"

# List of filenames to download from the model repository
filenames = [
    "config.json", "flax_model.msgpack", "generation_config.json", "model.safetensors",
    "pytorch_model.bin", "special_tokens_map.json", "spiece.model", "tf_model.h5",
    "tokenizer.json", "tokenizer_config.json"
]

# Attempt to download each file from the model repository
try:
    for filename in filenames:
        # Download the file from the Hugging Face Hub
        downloaded_model_hub = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            token=HUGGING_FACE_API_KEY
        )
        print(f"Downloaded {filename} to {downloaded_model_hub}")
except Exception as e:
    print(f"An error occurred: {e}")

# Import necessary classes from the transformers library
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model for the specified model identifier
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

# Define the input text for translation
input_text = "translate English to German: How old are you? I want to know what happened around here and need to eat good food."

# Tokenize the input text and convert it to tensor format
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate the translation output using the model
outputs = model.generate(input_ids)

# Decode the generated output tensor back to text
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the translated text
print(f"Translated Text: {translated_text}")

```
