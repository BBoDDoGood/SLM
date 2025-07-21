#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKO-T5-large 기반 군중 모니터링 SLM 모델 학습 코드
수학 추론 능력 향상을 위한 대용량 모델 사용
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

# 수학 추론 능력 향상을 위한 모델 옵션
MODEL_OPTIONS = [
    "paust/pko-t5-large",          # 한국어 T5 Large (1순위 - 수학 추론 우수)
    "google/flan-t5-large",         # Flan-T5 Large (수학 추론 특화)
    "google/mt5-large",             # 다국어 T5 Large
    "paust/pko-t5-base",            # 한국어 T5 Base (대안)
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

OUTPUT_DIR = "/Volumes/Data/slm_model_large"  # Large 모델용 별도 디렉토리
SEED = 42

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pko_t5_large_train_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset_with_math_focus(csv_files):
    """수학 계산이 포함된 데이터셋 로드 (수학 추론 능력 향상용)"""
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
        
        # 수학 계산이 포함된 데이터 필터링 (선택적)
        math_keywords = ['명', '기준', '수용인원', '허용', '최대', '제한', '초과', '부족', '차이']
        math_data = df[df['Input'].str.contains('|'.join(math_keywords), na=False)]
        logger.info(f"수학 계산 데이터 수: {len(math_data)}")
        
        # 수학 계산 데이터를 우선적으로 포함
        if len(math_data) > 0:
            # 수학 데이터 70%, 일반 데이터 30% 비율로 조정
            math_ratio = min(0.7, len(math_data) / len(df))
            math_sample_size = int(len(df) * math_ratio)
            general_sample_size = len(df) - math_sample_size
            
            # 수학 데이터 샘플링
            math_sample = math_data.sample(n=min(math_sample_size, len(math_data)), random_state=SEED)
            
            # 일반 데이터에서 수학 데이터 제외 후 샘플링
            general_data = df[~df.index.isin(math_data.index)]
            general_sample = general_data.sample(n=min(general_sample_size, len(general_data)), random_state=SEED)
            
            # 데이터 합치기
            df = pd.concat([math_sample, general_sample], ignore_index=True)
            logger.info(f"수학 계산 중심 데이터셋: {len(df)}개")
        
        # 수학 추론을 위한 입력 포맷 강화
        df['input_text'] = "수학 분석: " + df['Domain'] + ", " + df['Input']
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

def preprocess_function_math_focused(examples, tokenizer, model_type="t5"):
    """수학 추론에 특화된 전처리 함수"""
    try:
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        # T5 계열 모델의 경우 수학 분석 prefix 추가
        if "t5" in model_type.lower():
            inputs = ["수학 분석: " + text.replace("수학 분석: ", "") for text in inputs]
        
        # 입력 토큰화 (수학 추론을 위해 더 긴 시퀀스 허용)
        model_inputs = tokenizer(
            inputs,
            max_length=512,  # 수학 계산을 위해 더 긴 시퀀스
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # 타겟 토큰화
        labels = tokenizer(
            targets,
            max_length=512,  # 수학 계산 결과를 위해 더 긴 시퀀스
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # 패딩 토큰을 -100으로 설정
        model_inputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]
        
        return model_inputs
        
    except Exception as e:
        logger.error(f"전처리 중 오류: {e}")
        raise

def try_load_large_model():
    """Large 모델 로드 시도"""
    for model in MODEL_OPTIONS:
        try:
            logger.info(f"Large 모델 로드 시도: {model}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model)
            
            # 토크나이저 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"✅ Large 모델 로드 성공: {model}")
            logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model_obj.parameters()):,}")
            return model_obj, tokenizer, model
            
        except Exception as e:
            logger.warning(f"❌ Large 모델 로드 실패: {model}, 오류: {e}")
            continue
    
    raise ValueError("사용 가능한 Large 모델을 찾을 수 없습니다.")

def test_math_reasoning(model, tokenizer, device, model_name):
    """수학 추론 능력 테스트"""
    logger.info("🧮 수학 추론 능력 테스트...")
    
    math_test_cases = [
        "군중 밀집 및 체류 감지, 농구경기장 계단에 12:14 현재 4명이 있으며, 운영 기준은 9명입니다.",
        "군중 밀집 및 체류 감지, 13:13 클리닉에서 71명 밀집, 기준 수용인원 49명",
        "군중 밀집 및 체류 감지, 8:29 구청 309명 계측, 기준 인원 401명",
        "군중 밀집 및 체류 감지, 20:32 백화점 47명 파악, 기준 39명",
        "군중 밀집 및 체류 감지, 오전 11시 10분 극장 80명 계측, 기준 인원 63명"
    ]
    
    # T5 계열 모델의 경우 prefix 추가
    if "t5" in model_name.lower():
        math_test_cases = ["수학 분석: " + text for text in math_test_cases]
    
    model.eval()
    try:
        for i, input_text in enumerate(math_test_cases, 1):
            # 토큰화
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 생성 (수학 추론을 위해 더 정교한 설정)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,  # 더 정확한 생성을 위해 빔 서치 증가
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    temperature=0.7,  # 창의성과 정확성의 균형
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"수학 테스트 {i}:")
            logger.info(f"입력: {input_text}")
            logger.info(f"출력: {generated_text}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"수학 추론 테스트 중 오류: {e}")

def main():
    """메인 실행 함수 (Large 모델 수학 추론 특화)"""
    set_seed(SEED)
    
    # 디바이스 설정 (Large 모델은 GPU 권장)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")
    
    # Large 모델 로드
    logger.info("🤖 Large 모델 로드 중...")
    model, tokenizer, model_name = try_load_large_model()
    model = model.to(device)
    
    # 모델 정보 출력
    logger.info(f"선택된 Large 모델: {model_name}")
    logger.info(f"토크나이저 vocab 크기: {len(tokenizer)}")
    logger.info(f"패드 토큰: {tokenizer.pad_token}")
    
    # 수학 추론 중심 데이터셋 로드
    logger.info("📊 수학 추론 중심 데이터셋 로드 중...")
    dataset = load_dataset_with_math_focus(CSV_FILES)
    logger.info(f"데이터셋 로드 완료: 훈련 {len(dataset['train'])}개, 검증 {len(dataset['validation'])}개")
    
    # 데이터 토큰화
    logger.info("🔄 데이터 토큰화 중...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function_math_focused(x, tokenizer, model_name),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Large 모델에 최적화된 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,  # Large 모델에 적합한 낮은 학습률
        per_device_train_batch_size=2,  # Large 모델 메모리 제약
        per_device_eval_batch_size=2,
        num_train_epochs=2,  # Large 모델은 적은 에포크로도 충분
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True if device.type == "cuda" else False,  # GPU 사용시 혼합 정밀도
        gradient_checkpointing=True,  # 메모리 절약
        dataloader_pin_memory=False,  # GPU 멀티프로세싱 문제 해결
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # GPU 멀티프로세싱 문제 해결
        report_to=[],
        # Large 모델 최적화 설정
        max_grad_norm=1.0,
        optim="adamw_torch",
        seed=SEED,
        gradient_accumulation_steps=4,  # 효과적인 배치 크기 증가
        # 수학 추론을 위한 추가 설정
        generation_max_length=512,
        generation_num_beams=4
    )
    
    # 트레이너 설정
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 학습 시작
    logger.info(f"🚀 PKO-T5-Large 모델 수학 추론 학습 시작: {model_name}")
    start_time = datetime.datetime.now()
    
    # 체크포인트에서 재개 가능
    resume_from_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(OUTPUT_DIR, latest)
            logger.info(f"체크포인트에서 재개: {resume_from_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 모델 저장
    logger.info("💾 Large 모델 저장 중...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 수학 추론 능력 테스트
    logger.info("🧮 학습된 Large 모델 수학 추론 테스트...")
    test_math_reasoning(model, tokenizer, device, model_name)
    
    elapsed = datetime.datetime.now() - start_time
    logger.info(f"✅ Large 모델 학습 완료! 총 시간: {elapsed}")
    
    return True

if __name__ == "__main__":
    main()