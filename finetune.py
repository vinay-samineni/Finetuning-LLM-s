# dependency installing
pip install langchain_community
pip install transformers
pip install datasets
pip install peft

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

#dataset loading
from datasets import load_dataset
dataset = load_dataset("json",data_files="")#please give your own file path in between ""

# prompt template
def format_data(example):
    prompt_template = f'''[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense or is not factually coherent, explain why instead of answering incorrectly. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{example["question"]}[/INST]

{example["answer"]}
'''
    return {"formatted_prompt": prompt_template}

# Apply formatting to the dataset
formatted_dataset = dataset["train"].map(format_data)

# Display an example
print(formatted_dataset[0]["formatted_prompt"])

#login to hugging face using your token
huggingface-cli login

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Tokenization

def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["formatted_prompt"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Labels should be same as input_ids for CLM training
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

    return tokenized_inputs

# Assign a padding token before tokenizing
tokenizer.pad_token = tokenizer.eos_token

# Apply tokenization
tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True, remove_columns=formatted_dataset.column_names)

#Parametrized effiecient Finetuning and Low rank adaption
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

#Training parameters
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    per_device_train_batch_size=1,  # Reduce batch size (was 4)
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Helps stabilize training
    num_train_epochs=2,
    save_steps=500,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    fp16=True  # Mixed precision for lower memory usage
)

# train using wandb Weights and biases
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # Ideally, separate train and eval sets
)

# Start training
trainer.train()

#save the finetuned model
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")
print("✅ Model saved successfully!")

# code to infer the saved model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model_path = "./fine_tuned_llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define input prompt
input_text = "[INST] What is the purpose of the U.S. Constitution? [/INST] Answer: "

# Tokenize input and move to same device as the model
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

# Generate response
output = model.generate(
    **inputs,
    max_new_tokens=100,  # Avoids excessive length
    pad_token_id=tokenizer.eos_token_id,  # Handles padding properly
    do_sample=True,  # Enables random sampling
    temperature=0.7,  # Controls randomness
    top_p=0.9  # Keeps diversity in generation
)

# Decode response
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
generated_answer = decoded_output.split("[/INST]")[-1].strip()  # Extracts relevant part

# Print result
print(f"Question: What is the purpose of the U.S. Constitution?\nAnswer: {generated_answer}")
