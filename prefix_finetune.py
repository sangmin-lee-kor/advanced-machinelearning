import os
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import PrefixTuningConfig, get_peft_model
from datasets import Dataset
import fire

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # 레이블 추출
        outputs = model(**inputs)  # 모델 출력
        logits = outputs.logits

        # logits과 labels 크기 맞추기
        shift_logits = logits[:, :-1, :].contiguous()  # 마지막 토큰 제외
        shift_labels = labels[:, 1:].contiguous()  # 첫 번째 토큰 제외

        # logits과 labels 크기를 정확히 맞춤
        seq_length = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :seq_length, :]
        shift_labels = shift_labels[:, :seq_length]

        # CrossEntropyLoss를 위한 차원 변환
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = shift_labels.reshape(-1)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss

def train(
    data_path="train_data.json",  # 데이터 파일 경로
    base_model="yahma/llama-7b-hf",  # 베이스 모델 경로
    output_dir="./prefix_tuning_output",  # 출력 경로
    batch_size=4,  # 배치 사이즈
    num_epochs=3,  # 학습 에폭
    learning_rate=5e-5,  # 학습률
    gradient_accumulation_steps=4,  # 그라디언트 누적 단계
    max_input_length=512,  # 입력 최대 길이
    max_output_length=128,  # 출력 최대 길이
    eval_steps=100,  # 평가 간격
    save_steps=200,  # 저장 간격
    seed=42,  # 랜덤 시드
):
    # 데이터 로드
    def load_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    data = load_data(data_path)

    # 데이터 전처리
    def preprocess_data(data):
        inputs, outputs = [], []
        for item in data:
            inputs.append(f"Instruction: {item['instruction']}\nInput: {item['input']}\n")
            outputs.append(item['output'])
        return Dataset.from_dict({"input": inputs, "output": outputs})

    dataset = preprocess_data(data)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # 8비트 로드 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True  # 일부 연산을 CPU로 오프로드
    )

    # 모델과 토크나이저 로드
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정

    # Prefix Tuning 설정
    prefix_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        encoder_hidden_size=model.config.hidden_size,
    )

    model = get_peft_model(model, prefix_config)

    # 데이터 토큰화 함수
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"], max_length=max_input_length, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            examples["output"], max_length=max_output_length, truncation=True, padding="max_length"
        )
        # labels를 Tensor로 변환하여 -100으로 설정
        labels_tensor = torch.tensor(labels["input_ids"])
        labels_tensor[labels_tensor == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels_tensor.tolist()  # 다시 리스트로 변환
        return model_inputs

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # 데이터셋 포맷 변환
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        report_to="none",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 학습 시작
    trainer.train()

    # 학습된 모델 저장
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Prefix Tuning 완료 및 모델 저장 완료: {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
