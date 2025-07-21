#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKO-T5 기반 군중 모니터링 SLM 모델 학습 코드 (개선된 안정화 버전)
train_kobart_v2.py의 안정적인 패턴을 적용
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

# 모델 우선순위 (안정성 순)
MODEL_OPTIONS = [
    "paust/pko-t5-base",           # 한국어 T5 (1순위)
    "google/flan-t5-base",         # 대안 T5
    "google/mt5-small",            # 다국어 T5
    "facebook/mbart-large-50"      # 다국어 BART
]

CSV_FILES = [
    "/Users/yunseong/Desktop/SLM_Model/csv/domain1_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain2_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain3_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain4_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain5_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain6_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain7_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain8_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain9_dataset.csv",
    "/Users/yunseong/Desktop/SLM_Model/csv/domain10_dataset.csv",
]
OUTPUT_DIR = "/Volumes/Data/slm_model"
SEED = 42

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pko_t5_stable_train_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(csv_files):
    """CSV 파일들에서 데이터셋 로드 (안정화 버전)"""
    try:
        all_data = []
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
                logger.info(f"로드된 파일: {csv_file}, 데이터 수: {len(df)}")
            else:
                logger.warning(f"파일을 찾을 수 없음: {csv_file}")
        
        if len(all_data) == 0:
            raise ValueError("로드된 CSV 파일이 없습니다.")
        
        # 모든 데이터 합치기
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"원본 데이터 수: {len(df)}")
        
        # 결측값 제거
        df = df.dropna(subset=['Domain', 'Input', 'Output'])
        logger.info(f"정제된 데이터 수: {len(df)}")
        
        # 안정적인 입력 포맷 (train_kobart_v2.py 스타일)
        df['input_text'] = df['Domain'] + ", " + df['Input']
        df['target_text'] = df['Output']
        
        # 통계 정보
        logger.info(f"평균 입력 길이: {df['input_text'].str.len().mean():.1f}자")
        logger.info(f"평균 출력 길이: {df['target_text'].str.len().mean():.1f}자")
        logger.info(f"도메인 종류: {df['Domain'].unique().tolist()}")
        
        # 데이터 분할
        train_df, eval_df = train_test_split(
            df[['input_text', 'target_text']], 
            test_size=0.2, 
            random_state=SEED
        )
        
        # Dataset 객체 생성
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': eval_dataset
        })
        
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise

def preprocess_function_safe(examples, tokenizer, model_type="t5"):
    """안전한 전처리 함수 (train_kobart_v2.py 패턴)"""
    
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    # T5 계열 모델의 경우 간단한 prefix 추가
    if "t5" in model_type.lower():
        inputs = ["분석: " + text for text in inputs]
    
    try:
        # 입력 토큰화 (더 안전한 설정)
        model_inputs = tokenizer(
            inputs,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # 타겟 토큰화
        labels = tokenizer(
            targets,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # 패딩 토큰을 -100으로 설정 (표준 방법)
        model_inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]
        
        return model_inputs
        
    except Exception as e:
        logger.error(f"전처리 중 오류: {e}")
        # 최소한의 안전한 전처리
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors=None
        )
        
        labels = tokenizer(
            targets,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors=None
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def try_load_model(model_name):
    """모델 로드 시도 (train_kobart_v2.py 패턴)"""
    try:
        logger.info(f"모델 로드 시도: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 패드 토큰이 없으면 추가 (특수 토큰 추가 대신 안전한 방법)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"✅ 모델 로드 성공: {model_name}")
        return tokenizer, model, model_name
        
    except Exception as e:
        logger.warning(f"❌ 모델 로드 실패: {model_name} - {e}")
        return None, None, None

def test_model_safe(model, tokenizer, device, model_name):
    """안전한 모델 테스트"""
    logger.info("학습된 모델 테스트...")
    
    test_inputs = [
        "군중 밀집 및 체류 감지, 13:00 공연장 입구 외곽 감지 인원 55명",
        "군중 밀집 및 체류 감지, 현재 시각은 08시 15분이며, 도서관 2층 정문 앞에 18명이 있습니다. 기준 인원은 25명입니다."
    ]
    
    # T5 계열 모델의 경우 prefix 추가
    if "t5" in model_name.lower():
        test_inputs = ["분석: " + text for text in test_inputs]
    
    model.eval()
    try:
        for i, input_text in enumerate(test_inputs, 1):
            # 토큰화
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True,
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 생성
            with torch.no_grad():
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
            
            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"테스트 {i}:")
            logger.info(f"입력: {input_text}")
            logger.info(f"출력: {generated_text}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"테스트 중 오류: {e}")

def main():
    """메인 실행 함수 (안정화 버전)"""
    set_seed(SEED)
    
    # 디바이스 설정 (안정성 우선)
    device = "cpu"
    logger.info(f"사용 디바이스: {device}")
    
    # 모델 로드 시도 (우선순위 순)
    tokenizer, model, model_name = None, None, None
    
    for model_option in MODEL_OPTIONS:
        tokenizer, model, model_name = try_load_model(model_option)
        if model is not None:
            break
    
    if model is None:
        logger.error("사용 가능한 모델이 없습니다!")
        return False
    
    model = model.to(device)
    
    # 토크나이저 정보
    logger.info(f"선택된 모델: {model_name}")
    logger.info(f"토크나이저 vocab 크기: {len(tokenizer)}")
    logger.info(f"패드 토큰: {tokenizer.pad_token}")
    logger.info(f"EOS 토큰: {tokenizer.eos_token}")
    
    # 데이터셋 로드
    logger.info("데이터셋 로드 중...")
    dataset = load_dataset(CSV_FILES)
    
    # 데이터 토큰화
    logger.info("데이터 토큰화 중...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function_safe(x, tokenizer, model_name),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 학습 인수 설정 (train_kobart_v2.py의 안정적인 설정)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,  # 안전한 학습률
        per_device_train_batch_size=1,  # 작은 배치
        per_device_eval_batch_size=1,
        num_train_epochs=3,  # PKO-T5에 적합한 에포크
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=20,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=2,
        predict_with_generate=False,  # 안정성을 위해 비활성화
        fp16=False,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to=[],
        # 추가 안정성 설정
        max_grad_norm=1.0,
        optim="adamw_torch",
        seed=SEED,
        gradient_accumulation_steps=2  # 효과적인 배치 크기 증가
    )
    
    # 트레이너 설정 (Seq2SeqTrainer 사용)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 학습 시작
    logger.info(f"PKO-T5 모델 학습 시작: {model_name}")
    start_time = datetime.datetime.now()
    
    try:
        # 체크포인트 확인 및 재개 학습 지원
        resume_from_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
            if checkpoints:
                # 가장 최근 체크포인트 찾기
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_from_checkpoint = os.path.join(OUTPUT_DIR, latest_checkpoint)
                logger.info(f"🔄 기존 체크포인트에서 재개: {resume_from_checkpoint}")
            else:
                logger.info("🆕 새로운 학습 시작 (체크포인트 없음)")
        else:
            logger.info("🆕 새로운 학습 시작 (출력 디렉토리 없음)")
        
        # 학습 실행 (체크포인트가 있으면 재개, 없으면 새로 시작)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("학습 완료!")
        
        # 모델 저장
        logger.info("모델 저장 중...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"모델이 {OUTPUT_DIR}에 저장되었습니다.")
        
        # 간단 테스트
        test_model_safe(model, tokenizer, device, model_name)
        
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        logger.info(f"총 학습 시간: {training_time}")
        
        return True
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🚀 PKO-T5 군중 모니터링 모델 학습 시작 (안정화 버전)")
    logger.info("="*80)
    
    success = main()
    
    if success:
        logger.info("="*80)
        logger.info("✅ PKO-T5 모델 학습이 성공적으로 완료되었습니다!")
        logger.info("="*80)
    else:
        logger.info("="*80)
        logger.info("❌ PKO-T5 모델 학습 중 오류가 발생했습니다.")
        logger.info("="*80)