# 🦙 LLaMA Instruction Fine-Tuning with LoRA (PEFT)

This project demonstrates how to **fine-tune the Meta LLaMA 3.2-1B-Instruct** model using **instruction-style prompts** and **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning with Hugging Face's `transformers` and `peft` libraries. It also includes **inference code** to generate responses from the fine-tuned model.

---

## 📌 Features

- ✅ Instruction-style prompt formatting
- ✅ Dataset loading and preprocessing with Hugging Face `datasets`
- ✅ Tokenization and padding using `AutoTokenizer`
- ✅ Parameter-efficient fine-tuning with `peft` (LoRA)
- ✅ Training with `Trainer` from Hugging Face
- ✅ Model saving and loading
- ✅ Prompt-based response generation from the fine-tuned model

---

## 📁 Project Structure

llama-lora-finetune/
├── train.py # Complete fine-tuning pipeline
├── infer.py # Inference script for fine-tuned model
├── data/
│ └── your_dataset.json # Input training dataset (custom format)
├── fine_tuned_llama/ # Saved model and tokenizer
├── README.md # Documentation
└── requirements.txt # Dependencies


---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/llama-lora-finetune.git
cd llama-lora-finetune

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Installing dependencies
 all the dependencies are listed in code itself

📂 Input Dataset Format
Provide your dataset as a JSON file with at least the following format:
[
  {
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris."
  },
  ...
]

Save it as data/train.json or update the path accordingly in your script.

🧠 Fine-Tuning Script Overview (train.py)

Load dataset using datasets.load_dataset
Format data using instruction-based prompts with [INST]...[/INST]
Tokenize inputs and apply LoRA via peft
Train using Hugging Face's Trainer
Save the fine-tuned model and tokenizer

🏁 Run Training
python train.py

The model will be saved to ./fine_tuned_llama/

🤖 Inference Script Overview (infer.py)
Loads the fine-tuned model and uses a manually crafted prompt to generate an answer.

🔍 Example Prompt
[INST] What is the purpose of the U.S. Constitution? [/INST]

▶️ Run Inference
python infer.py

Question: What is the purpose of the U.S. Constitution?
Answer: The U.S. Constitution establishes the framework of the federal government and outlines the rights of citizens.

📦 Output

./fine_tuned_llama/: Contains the trained model and tokenizer
./logs/: Training logs (optional)
output: Printed responses after inference

🧪 Tips

Adjust max_length and top_p during generation for different response styles.
Replace train_dataset and eval_dataset with split datasets for better generalization.
You can use wandb or tensorboard for monitoring training progress.

Let me know if you want:

train.py and infer.py modularized as files
A web UI on top of this fine-tuned model (e.g., Streamlit or Flask)
Help uploading the model to Hugging Face Hub
I'm happy to assist!
