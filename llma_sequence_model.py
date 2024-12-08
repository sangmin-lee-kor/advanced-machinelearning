from transformers import TrainingArguments, Trainer, LlamaForCausalLM, LlamaTokenizer, DataCollatorForSeq2Seq
from peft import get_peft_config, LoraConfig, get_peft_model, get_peft_model_state_dict
import torch
from typing import List
import fire
import sys
from datasets import load_dataset
import os

ACCESS_TOKEN = "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"

def train(
    # model/data params
    base_model: str = "openlm-research/open_llama_7b",  # the only required argument
    data_path: str = "data/train",
    output_dir: str = "checkpoint",
    # training hyperparams
    batch_size: int = 128,#used to be 128
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size


    # 사전 학습 모델 로드
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16, use_auth_token=ACCESS_TOKEN
        )
    tokenizer = LlamaTokenizer.from_pretrained(base_model, use_auth_token=ACCESS_TOKEN, tokenizer_class=LlamaTokenizer,)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    lora_config = LoraConfig(
    r=lora_r,                      # Low-rank 차원
    lora_alpha=lora_alpha,            # Scaling factor
    target_modules=lora_target_modules,  # 적용할 모듈 (모델 구조에 따라 다름)
    lora_dropout=lora_dropout,         # Dropout 비율
    bias="none",              # Bias 처리 방식
    task_type="CAUSAL_LM"     # 작업 유형 (예: CAUSAL_LM, SEQ_2_SEQ_LM 등)
    )

    # LoRA를 모델에 적용
    model = get_peft_model(model, lora_config)

    def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    data_set = load_dataset("json", data_files=data_path)

    split_data = data_set.shuffle().map(lambda prompt: generate_and_tokenize_prompt(prompt, tokenizer, cutoff_len))
    train_data = split_data["train"]
    validation_data = None

    ##### Train
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(output_dir)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""### Instruction: 
                {data_point["instruction"]}
                ### input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    

if __name__ == "__main__":
    fire.Fire(train)