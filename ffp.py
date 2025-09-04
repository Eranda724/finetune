import os

# Disable optional backends in Transformers to avoid importing TensorFlow/Keras
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sklearn.model_selection import train_test_split
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=None
)
dataset = load_dataset("noorulhasan/Chatbot_QnA")
print(f"Dataset: {len(dataset['train'])} samples")
df = dataset["train"]
train_indices, val_indices = train_test_split(
    range(len(df)), 
    test_size=0.1, 
    random_state=42
)
train_df = df.select(train_indices)
val_df = df.select(val_indices)
def format_qa(example):
    return {
        "text": f"Q: {example['Question']}\nA: {example['Answer']}{tokenizer.eos_token}"
    }
train_df = train_df.map(format_qa)
val_df = val_df.map(format_qa)
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None
    )
    
    result["labels"] = result["input_ids"].copy()
    return result
train_ds = train_df.map(tokenize_function, batched=True, remove_columns=["text", "Question", "Answer"])
val_ds = val_df.map(tokenize_function, batched=True, remove_columns=["text", "Question", "Answer"])
print(f"Training: {len(train_ds)}, Validation: {len(val_ds)}")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
output_dir = "./lora-gpt2-qa"
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to="none",
    fp16=False,
    dataloader_drop_last=False,
    remove_unused_columns=False,
    prediction_loss_only=True,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)
print("Starting training...")
trainer.train()
model.save_pretrained(f"{output_dir}/adapter")
tokenizer.save_pretrained(f"{output_dir}/adapter")
print(f"Model saved!")
def generate_answer(question, max_new_tokens=64, temperature=0.6, top_p=0.85):
    device = next(model.parameters()).device
    model.eval()
    
    prompt = f"Human: {question}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    answer = generated_text.split("Human:")[0].strip()
    return answer if answer else generated_text.strip()
def test_model(questions):
    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Test {i}]")
        print(f"Q: {question}")
        answer = generate_answer(question)
        print(f"A: {answer}")
        print("-" * 40)
test_questions = [
    "Who created the Hebbian learning rule?",
    "When was the first neural network built?",
    "What is machine learning?",
    "How does backpropagation work?",
    "What is the difference between AI and ML?",
    "What is deep learning?",
]
test_model(test_questions)
def load_model_for_inference(adapter_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    return model, tokenizer