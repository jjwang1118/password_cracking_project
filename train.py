from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import yaml
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from util.prompt_template import _get_indice
from util.tokenize import process_train_targeted, get_alpha_vocab
from pathlib import Path
import torch

if __name__ == "__main__":

    # 載入參數
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    prompt_idx=config["train"]["prompt_template_id"]
    train_config=config["train"]
    lora=config["train"]["lora_config"]

    # 訓練及和測試及
    project_root = Path(__file__).parent
    train_path = project_root / config["dataset_path"] / "split" / "train_data.jsonl"
    test_path  = project_root / config["dataset_path"] / "split" / "test_data.jsonl"

    datasets=load_dataset(
        "json",
        data_files={
            "train":str(train_path),
            "test":str(test_path)
        }
    )

    # 更改prompt template
    prompt_template = _get_indice(prompt_idx)

    model_name = train_config["train_config"]["model_name"]
    local_model_path = Path("model") / model_name
    model_source = str(local_model_path) if local_model_path.exists() else model_name

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForCausalLM.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab = get_alpha_vocab(tokenizer)

    preprocess_fn = partial(
        process_train_targeted,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        vocab=vocab,
        max_length=512,
    )

    train_dataset = datasets["train"].map(
        preprocess_fn,
        batched=False,
        remove_columns=datasets["train"].column_names,
    )
    eval_dataset = datasets["test"].map(
        preprocess_fn,
        batched=False,
        remove_columns=datasets["test"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        padding=True,            

    )
    
    # lora
    lora_config=LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        init_lora_weights=lora["init_lora_weights"],
        target_modules=lora["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)


    #模型
    training_args=TrainingArguments(
        output_dir=train_config["train_config"]["output_dir"]+train_config["train_config"]["model_name"],
        num_train_epochs=train_config["train_config"]["num_train_epochs"],
        per_device_train_batch_size=train_config["train_config"]["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["train_config"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["train_config"]["gradient_accumulation_steps"],
        eval_strategy="steps",
        eval_steps=train_config["train_config"]["eval_steps"],
        save_strategy="steps",
        save_steps=train_config["train_config"]["save_steps"],
        logging_steps=train_config["train_config"]["logging_steps"],
        load_best_model_at_end=True,
        save_total_limit=5,
        logging_dir=f"./logs/{train_config['train_config']['model_name']}",
        report_to="tensorboard",
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        seed=config["seed"],
        learning_rate=train_config["train_config"]["learning_rate"],
        weight_decay=train_config["train_config"]["weight_decay"],
        warmup_ratio=train_config["train_config"]["warmup_ratio"],
        optim=train_config["train_config"]["optim"],
        eval_on_start=True,            
        label_smoothing_factor=0,        
        data_seed=config["seed"],                     
        push_to_hub=False,       
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    trainer.train()

    lora_save_path = Path(train_config["train_config"]["output_dir"]) / model_name / "lora_final"
    model.save_pretrained(str(lora_save_path))
    

    
