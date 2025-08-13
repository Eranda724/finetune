# lora_gpt2_demo.py  (minimal, for practice)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

model_name = "gpt2"

# 1) load
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # needed when batching with Trainer
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2) tiny dataset (replace with your data)
texts = [
    "Q: What is the capital of France?\nA: Paris.",
    "Q: How do you say hello in Spanish?\nA: hola.",
    "Q: Who wrote Hamlet?\nA: William Shakespeare."
]
ds = Dataset.from_dict({"text": texts})

# 3) tokenize
def tok(example):
    return tokenizer(example["text"], truncation=True, max_length=128, padding="max_length")
ds = ds.map(tok, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 4) LoRA config (simple defaults)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn","c_proj"],   # common for GPT-2; change per-model if needed
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())  # see how few params are trainable

# 5) train (very small settings to run quickly)
training_args = TrainingArguments(
    output_dir="./lora-gpt2",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,          # log every step (more frequent updates)
    save_strategy="no",
    report_to="none",        # disable wandb or other trackers for cleaner output
    disable_tqdm=False       # enable the tqdm progress bar in terminal
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
trainer.train()

# 6) save adapter only
model.save_pretrained("lora-gpt2-adapter")