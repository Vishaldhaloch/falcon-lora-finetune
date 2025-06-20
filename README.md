# falcon-lora-finetune
# 🦅 Falcon LoRA - IMDb Sentiment Fine-Tuning

This repository contains the code for fine-tuning the [tiiuae/falcon-rw-1b](https://huggingface.co/tiiuae/falcon-rw-1b) model using the [IMDb movie reviews dataset](https://huggingface.co/datasets/imdb), leveraging the LoRA (Low-Rank Adaptation) method from PEFT for efficient training.

## 📂 Project Structure

modelfinetuning/
├── train.py # Fine-tuning script
├── test_lora_inference.py # Inference script using trained adapter
├── pushtohuggingface.ipynb # Notebook to push LoRA adapter to Hugging Face Hub
├── requirements.txt # All required dependencies
└── README.md # Project overview and instructions



---

## 🚀 Model Overview

- **Base Model:** `tiiuae/falcon-rw-1b`
- **Fine-tuning Method:** LoRA via `peft`
- **Dataset:** IMDb (1000 samples for demo)
- **Training Frameworks:** 🤗 Transformers + PEFT
- **Trained On:** Google Colab (T4 GPU)
- **Inference Pipeline:** Text Generation (Sentiment-based completion)

🧠 **Final Model on Hugging Face Hub:**
➡️ [vishal1d/falcon-lora-imdb](https://huggingface.co/vishal1d/falcon-lora-imdb)

---

## 📦 Requirements

Install dependencies using:

pip install -r requirements.txt

## 🏋️‍♂️ Training (LoRA)
python train.py
## Inference (with LoRA)
python test_lora_inference.py
## This loads your trained adapter and runs a test prompt:

prompt = "The movie was absolutely wonderful because"

## Output:
Generated Output:
The movie was absolutely wonderful because...


## ☁️ Push to Hugging Face Hub
To upload your adapter to your Hugging Face account:

Login:

from huggingface_hub import notebook_login
notebook_login()

## Run
!python pushtohuggingface.ipynb








