#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAN-T5 기반 군중 모니터링 SLM 모델 학습 코드
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrowdMonitoringDataset(Dataset):
    """군중 모니터링 데이터셋 클래스"""
    
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
        
        # 입력과 타겟 토큰화
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
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def load_and_preprocess_data(csv_files):
    """CSV 파일들을 로드하고 전처리"""
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
    
    # 입력 텍스트 생성 (Domain + Input)
    inputs = []
    for _, row in combined_df.iterrows():
        input_text = f"도메인: {row['Domain']}, 입력: {row['Input']}"
        inputs.append(input_text)
    
    targets = combined_df['Output'].tolist()
    
    logger.info(f"전체 데이터 수: {len(inputs)}")
    
    return inputs, targets

def main():
    # 설정 (더 작은 모델 사용)
    MODEL_NAME = "google/flan-t5-small"  # base 대신 small 모델 사용
    OUTPUT_DIR = "./flant5-crowd-monitoring"
    CSV_FILES = [
        "../csv/domain1_dataset - 시트1.csv",
        "../csv/domain1_example_dataset.csv"
    ]
    
    # 하이퍼파라미터 (개선된 설정)
    BATCH_SIZE = 2  # 배치 크기 증가
    LEARNING_RATE = 3e-4  # 학습률 증가
    NUM_EPOCHS = 5  # 에포크 수 증가
    MAX_LENGTH = 512  # 최대 길이 복원
    
    # 디바이스 설정 (CPU 강제 사용)
    device = torch.device('cpu')  # MPS 메모리 부족으로 CPU 사용
    logger.info(f"사용 디바이스: {device}")
    
    # 메모리 최적화 설정
    import os
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # 데이터 로드 및 전처리
    logger.info("데이터 로드 중...")
    inputs, targets = load_and_preprocess_data(CSV_FILES)
    
    # 데이터 분할
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )
    
    logger.info(f"훈련 데이터: {len(train_inputs)}, 검증 데이터: {len(val_inputs)}")
    
    # 토크나이저 및 모델 로드
    logger.info("모델 및 토크나이저 로드 중...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # 특수 토큰 추가 (필요시)
    special_tokens = ["<도메인>", "<입력>", "<출력>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
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
        padding=True
    )
    
    # 훈련 인자 설정
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
        report_to=None,  # 텐서보드 비활성화
        dataloader_pin_memory=False,
        gradient_accumulation_steps=4,  # 그래디언트 누적 조정
        fp16=False,  # FP16 비활성화
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
    logger.info("모델 학습 시작...")
    start_time = datetime.datetime.now()
    
    trainer.train()
    
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    logger.info(f"학습 완료! 소요 시간: {training_time}")
    
    # 모델 저장
    logger.info("모델 저장 중...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 학습 로그 저장
    log_file = f"flant5_train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    logger.info(f"학습 로그가 {log_file}에 저장되었습니다.")
    logger.info(f"모델이 {OUTPUT_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    main() 