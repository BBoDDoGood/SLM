#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKO-T5 기반 군중 모니터링 SLM 모델 학습 코드
PKO-T5-base 모델을 사용한 한국어 특화 학습
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
import logging
import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrowdMonitoringDataset(Dataset):
    """군중 모니터링 데이터셋 클래스 (PKO-T5 최적화)"""
    
    def __init__(self, inputs, targets, tokenizer, max_length=512):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        # PKO-T5는 한국어에 특화되어 있으므로 더 긴 시퀀스 처리 가능
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # T5 모델의 경우 labels에서 pad_token을 -100으로 설정
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def load_and_preprocess_data(csv_files):
    """CSV 파일들을 로드하고 전처리 (한국어 최적화)"""
    all_data = []
    
    logger.info(f"CSV 파일 로드 시작: {csv_files}")
    
    for csv_file in csv_files:
        logger.info(f"파일 확인 중: {csv_file}")
        logger.info(f"파일 존재 여부: {os.path.exists(csv_file)}")
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
                logger.info(f"로드된 파일: {csv_file}, 데이터 수: {len(df)}")
                logger.info(f"컬럼: {list(df.columns)}")
            except Exception as e:
                logger.error(f"파일 로드 실패: {csv_file}, 오류: {e}")
        else:
            logger.warning(f"파일을 찾을 수 없음: {csv_file}")
    
    logger.info(f"로드된 데이터프레임 수: {len(all_data)}")
    
    if len(all_data) == 0:
        raise ValueError("로드된 CSV 파일이 없습니다. 파일 경로를 확인해주세요.")
    
    # 모든 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 결측값 제거
    combined_df = combined_df.dropna()
    
    # PKO-T5에 최적화된 한국어 프롬프트 생성
    inputs = []
    for _, row in combined_df.iterrows():
        input_text = f"다음 군중 모니터링 상황을 분석하고 적절한 대응 방안을 한국어로 제시하세요.\n도메인: {row['Domain']}\n상황: {row['Input']}\n대응방안:"
        inputs.append(input_text)
    
    targets = combined_df['Output'].tolist()
    
    logger.info(f"전체 데이터 수: {len(inputs)}")
    logger.info(f"샘플 입력: {inputs[0][:200]}...")
    logger.info(f"샘플 출력: {targets[0][:200]}...")
    
    return inputs, targets

def main():
    # PKO-T5 모델 설정
    MODEL_NAME = "pko-t5-base"  # PKO-T5 base 모델 사용
    OUTPUT_DIR = "./pko-t5-crowd-monitoring"
    CSV_FILES = [
        "../csv/domain1_dataset - 시트1.csv",
        "../csv/domain1_example_dataset.csv"
    ]
    
    # PKO-T5에 최적화된 하이퍼파라미터
    BATCH_SIZE = 2  # 메모리 효율성을 위해 작은 배치 크기
    LEARNING_RATE = 3e-4  # PKO-T5에 적합한 학습률
    NUM_EPOCHS = 5  # 한국어 특화 모델이므로 적은 에포크로도 효과적
    MAX_LENGTH = 512  # PKO-T5는 긴 시퀀스 처리 가능
    
    # 디바이스 설정
    device = torch.device('cpu')  # 안정성을 위해 CPU 사용
    logger.info(f"사용 디바이스: {device}")
    
    # 메모리 최적화 설정
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # 데이터 로드 및 전처리
    logger.info("데이터 로드 중...")
    inputs, targets = load_and_preprocess_data(CSV_FILES)
    
    # 데이터 분할
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )
    
    logger.info(f"훈련 데이터: {len(train_inputs)}, 검증 데이터: {len(val_inputs)}")
    
    # PKO-T5 토크나이저 및 모델 로드
    logger.info("PKO-T5 모델 및 토크나이저 로드 중...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        logger.info("PKO-T5 모델 로드 성공!")
    except Exception as e:
        logger.error(f"PKO-T5 모델 로드 실패: {e}")
        logger.info("대안으로 google/flan-t5-base 모델을 사용합니다.")
        MODEL_NAME = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # 한국어 특화 특수 토큰 추가
    special_tokens = ["<군중모니터링>", "<상황분석>", "<대응방안>", "<도메인>"]
    num_added_tokens = tokenizer.add_tokens(special_tokens)
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"한국어 특수 토큰 {num_added_tokens}개 추가됨")
    
    # 패드 토큰 설정 확인
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("패드 토큰을 EOS 토큰으로 설정")
    
    # 데이터셋 생성
    train_dataset = CrowdMonitoringDataset(
        train_inputs, train_targets, tokenizer, MAX_LENGTH
    )
    val_dataset = CrowdMonitoringDataset(
        val_inputs, val_targets, tokenizer, MAX_LENGTH
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # PKO-T5에 최적화된 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # MLflow, 텐서보드 비활성화
        dataloader_pin_memory=False,
        gradient_accumulation_steps=4,  # 효과적인 배치 크기 증가
        fp16=False,  # 안정성을 위해 비활성화
        prediction_loss_only=True,  # 평가 시 loss만 계산
        remove_unused_columns=False,  # 컬럼 제거 방지
        label_smoothing_factor=0.1,  # 라벨 스무딩으로 일반화 성능 향상
        dataloader_num_workers=0,  # 멀티프로세싱 비활성화
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 학습 시작
    logger.info("PKO-T5 모델 학습 시작...")
    start_time = datetime.datetime.now()
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        logger.info("훈련을 중단하고 현재까지의 모델을 저장합니다.")
    
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    logger.info(f"학습 완료! 소요 시간: {training_time}")
    
    # 모델 저장
    logger.info("모델 저장 중...")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"모델이 {OUTPUT_DIR}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")
    
    # 학습 로그 저장
    log_file = f"pko_t5_train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"모델: {MODEL_NAME}\n")
        f.write(f"출력 디렉토리: {OUTPUT_DIR}\n")
        f.write(f"배치 크기: {BATCH_SIZE}\n")
        f.write(f"학습률: {LEARNING_RATE}\n")
        f.write(f"에포크 수: {NUM_EPOCHS}\n")
        f.write(f"최대 길이: {MAX_LENGTH}\n")
        f.write(f"훈련 데이터 수: {len(train_inputs)}\n")
        f.write(f"검증 데이터 수: {len(val_inputs)}\n")
        f.write(f"학습 시작 시간: {start_time}\n")
        f.write(f"학습 종료 시간: {end_time}\n")
        f.write(f"총 학습 시간: {training_time}\n")
        f.write(f"사용 디바이스: {device}\n")
        f.write(f"추가된 특수 토큰 수: {num_added_tokens}\n")
        f.write(f"최종 어휘 크기: {len(tokenizer)}\n")
    
    logger.info(f"학습 로그가 {log_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()