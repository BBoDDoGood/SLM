#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKO-T5 기반 군중 모니터링 SLM 모델 학습 코드 (Colab GPU 최적화 버전)
"""

import os
import logging
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import datetime

# ===========================
# ✅ Colab T4-GPU 환경 설정
# ===========================

MODEL_OPTIONS = [
    "paust/pko-t5-base",  # 1순위
    "google/flan-t5-base",
    "google/mt5-small",
    "facebook/mbart-large-50"
]

CSV_FILES = [
    "/content/drive/MyDrive/SLM/csv/domain1_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain2_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain3_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain4_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain5_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain6_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain7_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain8_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain9_dataset.csv",
    "/content/drive/MyDrive/SLM/csv/domain10_dataset.csv",
]

OUTPUT_DIR = "/content/drive/MyDrive/SLM/slm_output"
SEED = 42

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pko_t5_colab_gpu_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(csv_files):
    all_data = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_data.append(df)
            logger.info(f"로드된 파일: {csv_file}, 데이터 수: {len(df)}")
        else:
            logger.warning(f"파일 없음: {csv_file}")
    if len(all_data) == 0:
        raise ValueError("로드할 CSV 파일이 없습니다.")
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"총 데이터: {len(df)}")
    df = df.dropna(subset=['Domain', 'Input', 'Output'])
    logger.info(f"결측치 제거 후: {len(df)}")
    df['input_text'] = df['Domain'] + ", " + df['Input']
    df['target_text'] = df['Output']
    logger.info(f"도메인 종류: {df['Domain'].unique().tolist()}")
    train_df, eval_df = train_test_split(df[['input_text', 'target_text']], test_size=0.2, random_state=SEED)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    return DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset
    })

def preprocess_function_safe(examples, tokenizer, model_type="t5"):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    if "t5" in model_type.lower():
        inputs = ["분석: " + text for text in inputs]
    model_inputs = tokenizer(
        inputs, max_length=256, padding="max_length",
        truncation=True, return_tensors=None
    )
    labels = tokenizer(
        targets, max_length=256, padding="max_length",
        truncation=True, return_tensors=None
    )
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels["input_ids"]
    ]
    return model_inputs

def try_load_model(model_name):
    try:
        logger.info(f"모델 로드 시도: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✅ 모델 로드 성공: {model_name}")
        return tokenizer, model, model_name
    except Exception as e:
        logger.warning(f"❌ 모델 로드 실패: {model_name} - {e}")
        return None, None, None

def test_model_safe(model, tokenizer, device, model_name):
    logger.info("학습된 모델 테스트...")
    test_inputs = [
        "군중 밀집 및 체류 감지, 13:00 공연장 입구 외곽 감지 인원 55명",
        "군중 밀집 및 체류 감지, 현재 시각은 08시 15분이며, 도서관 2층 정문 앞에 18명이 있습니다."
    ]
    if "t5" in model_name.lower():
        test_inputs = ["분석: " + text for text in test_inputs]
    model.eval()
    with torch.no_grad():
        for i, input_text in enumerate(test_inputs, 1):
            inputs = tokenizer(
                input_text, return_tensors="pt",
                max_length=256, truncation=True, padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"입력: {input_text}")
            logger.info(f"출력: {generated}")
            logger.info("-" * 50)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")

    tokenizer, model, model_name = None, None, None
    for model_option in MODEL_OPTIONS:
        tokenizer, model, model_name = try_load_model(model_option)
        if model:
            break
    if model is None:
        logger.error("❌ 모델 로드 실패")
        return False

    model.to(device)

    logger.info(f"선택된 모델: {model_name}")
    logger.info(f"Vocab 크기: {len(tokenizer)}")
    logger.info(f"Pad Token: {tokenizer.pad_token}")

    dataset = load_dataset(CSV_FILES)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function_safe(x, tokenizer, model_name),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=200,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        fp16=True,  # ✅ Colab GPU 혼합 정밀도 활성화
        gradient_checkpointing=True,  # ✅ GPU 메모리 절약
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        push_to_hub=False,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=2,
        optim="adamw_torch"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("🚀 Colab GPU 학습 시작")
    start_time = datetime.datetime.now()

    resume_from_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(OUTPUT_DIR, latest)
            logger.info(f"체크포인트에서 재개: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("✅ 학습 완료")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    test_model_safe(model, tokenizer, device, model_name)
    elapsed = datetime.datetime.now() - start_time
    logger.info(f"총 학습 시간: {elapsed}")
    return True

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🚀 PKO-T5 군중 모니터링 Colab GPU 학습 시작")
    logger.info("="*80)
    success = main()
    if success:
        logger.info("="*80)
        logger.info("✅ 학습 성공")
        logger.info("="*80)
    else:
        logger.info("="*80)
        logger.info("❌ 학습 실패")
        logger.info("="*80)
