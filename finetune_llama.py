from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

model_name = "unsloth/Llama-3.2-3B-bnb-4bit"
max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files="pamap2_processed.jsonl")
train_dataset = dataset["train"]

training_args = TrainingArguments(
    output_dir="./llama3_2_3b_pamap2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    warmup_steps=50,
    optim="adamw_8bit",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

trainer.train()

model.save_pretrained("./llama3_2_3b_pamap2_finetuned")
tokenizer.save_pretrained("./llama3_2_3b_pamap2_finetuned")

FastLanguageModel.for_inference(model)
input_text = "Hand Accelerometer X: 0.5, Y: 1.2, Z: -0.3; Hand Gyroscope X: 0.1, Y: 0.2, Z: 0.0 -> Activity:"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))