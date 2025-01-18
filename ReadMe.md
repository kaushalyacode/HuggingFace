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

