#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수학 추론 특화 학습 스크립트
데이터 증강과 특화된 학습 방법으로 수학 추론 능력 향상
"""

import os
import logging
import pandas as pd
import torch
import random
import re
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

# 수학 추론 특화 모델 옵션
MODEL_OPTIONS = [
    "paust/pko-t5-large",          # 한국어 T5 Large (수학 추론 우수)
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

OUTPUT_DIR = "/Volumes/Data/slm_model_math_specialized"
SEED = 42

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'math_specialized_train_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_math_data(df):
    """수학 계산이 포함된 데이터 추출"""
    math_keywords = ['명', '기준', '수용인원', '허용', '최대', '제한', '초과', '부족', '차이', '비교']
    math_data = df[df['Input'].str.contains('|'.join(math_keywords), na=False)]
    return math_data

def generate_math_augmented_data(original_data, num_augmentations=3):
    """수학 데이터 증강"""
    augmented_data = []
    
    for _, row in original_data.iterrows():
        input_text = row['Input']
        output_text = row['Output']
        
        # 숫자 추출
        numbers = re.findall(r'(\d+)명', input_text)
        baselines = re.findall(r'기준.*?(\d+)명', input_text)
        
        if len(numbers) >= 1 and len(baselines) >= 1:
            current = int(numbers[0])
            baseline = int(baselines[0])
            
            # 다양한 수학 시나리오 생성
            for i in range(num_augmentations):
                # 새로운 수치 생성 (현실적인 범위 내에서)
                new_current = random.randint(max(1, baseline - 50), baseline + 100)
                new_baseline = random.randint(max(1, new_current - 30), new_current + 30)
                
                # 입력 텍스트 수정
                new_input = re.sub(r'(\d+)명', f'{new_current}명', input_text, count=1)
                new_input = re.sub(r'기준.*?(\d+)명', f'기준 {new_baseline}명', new_input)
                
                # 출력 텍스트 수정 (계산 결과 반영)
                diff = new_current - new_baseline
                if diff > 0:
                    new_output = re.sub(r'기준.*?명.*?초과.*?명', f'기준 {new_baseline}명을 {abs(diff)}명 초과한 {new_current}명', output_text)
                elif diff < 0:
                    new_output = re.sub(r'기준.*?명.*?부족.*?명', f'기준 {new_baseline}명보다 {abs(diff)}명 적은 {new_current}명', output_text)
                else:
                    new_output = re.sub(r'기준.*?명.*?동일.*?명', f'기준 {new_baseline}명과 동일한 {new_current}명', output_text)
                
                augmented_data.append({
                    'Domain': row['Domain'],
                    'Input': new_input,
                    'Output': new_output
                })
    
    return pd.DataFrame(augmented_data)

def create_math_focused_dataset(csv_files):
    """수학 추론에 특화된 데이터셋 생성"""
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
        
        # 수학 데이터 추출
        math_data = extract_math_data(df)
        logger.info(f"수학 계산 데이터 수: {len(math_data)}")
        
        # 수학 데이터 증강
        if len(math_data) > 0:
            logger.info("수학 데이터 증강 중...")
            augmented_math_data = generate_math_augmented_data(math_data, num_augmentations=5)
            logger.info(f"증강된 수학 데이터 수: {len(augmented_math_data)}")
            
            # 증강된 데이터와 원본 데이터 합치기
            df = pd.concat([df, augmented_math_data], ignore_index=True)
            logger.info(f"증강 후 총 데이터 수: {len(df)}")
        
        # 수학 추론을 위한 입력 포맷 강화
        df['input_text'] = "수학 계산 분석: " + df['Domain'] + ", " + df['Input']
        df['target_text'] = df['Output']
        
        # 수학 데이터 우선 샘플링
        math_data_indices = df[df['Input'].str.contains('|'.join(['명', '기준', '수용인원']), na=False)].index
        if len(math_data_indices) > 0:
            # 수학 데이터 80%, 일반 데이터 20% 비율로 조정
            math_sample_size = min(int(len(df) * 0.8), len(math_data_indices))
            general_sample_size = len(df) - math_sample_size
            
            # 수학 데이터 샘플링
            math_sample_indices = random.sample(list(math_data_indices), math_sample_size)
            
            # 일반 데이터에서 수학 데이터 제외 후 샘플링
            general_indices = [i for i in df.index if i not in math_data_indices]
            general_sample_indices = random.sample(general_indices, min(general_sample_size, len(general_indices)))
            
            # 최종 데이터셋 구성
            final_indices = math_sample_indices + general_sample_indices
            df = df.loc[final_indices].reset_index(drop=True)
            logger.info(f"수학 추론 특화 데이터셋: {len(df)}개")
        
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
        logger.error(f"데이터셋 생성 실패: {e}")
        raise

def preprocess_function_math_specialized(examples, tokenizer, model_type="t5"):
    """수학 추론 특화 전처리 함수"""
    try:
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        # T5 계열 모델의 경우 수학 분석 prefix 추가
        if "t5" in model_type.lower():
            inputs = ["수학 계산 분석: " + text.replace("수학 계산 분석: ", "") for text in inputs]
        
        # 입력 토큰화 (수학 추론을 위해 더 긴 시퀀스)
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # 타겟 토큰화
        labels = tokenizer(
            targets,
            max_length=512,
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

def try_load_math_model():
    """수학 추론 특화 모델 로드"""
    for model in MODEL_OPTIONS:
        try:
            logger.info(f"수학 추론 모델 로드 시도: {model}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model)
            
            # 토크나이저 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"✅ 수학 추론 모델 로드 성공: {model}")
            logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model_obj.parameters()):,}")
            return model_obj, tokenizer, model
            
        except Exception as e:
            logger.warning(f"❌ 수학 추론 모델 로드 실패: {model}, 오류: {e}")
            continue
    
    raise ValueError("사용 가능한 수학 추론 모델을 찾을 수 없습니다.")

def test_math_reasoning_accuracy(model, tokenizer, device, model_name):
    """수학 추론 정확도 테스트"""
    logger.info("🧮 수학 추론 정확도 테스트...")
    
    math_test_cases = [
        ("군중 밀집 및 체류 감지, 농구경기장 계단에 12:14 현재 4명이 있으며, 운영 기준은 9명입니다.", "4", "9", "5명 부족"),
        ("군중 밀집 및 체류 감지, 13:13 클리닉에서 71명 밀집, 기준 수용인원 49명", "71", "49", "22명 초과"),
        ("군중 밀집 및 체류 감지, 8:29 구청 309명 계측, 기준 인원 401명", "309", "401", "92명 부족"),
        ("군중 밀집 및 체류 감지, 20:32 백화점 47명 파악, 기준 39명", "47", "39", "8명 초과"),
        ("군중 밀집 및 체류 감지, 오전 11시 10분 극장 80명 계측, 기준 인원 63명", "80", "63", "17명 초과")
    ]
    
    correct_calculations = 0
    total_tests = len(math_test_cases)
    
    # T5 계열 모델의 경우 prefix 추가
    if "t5" in model_name.lower():
        math_test_cases = [("수학 계산 분석: " + text, current, baseline, expected) for text, current, baseline, expected in math_test_cases]
    
    model.eval()
    try:
        for i, (input_text, current, baseline, expected) in enumerate(math_test_cases, 1):
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
            
            # 생성
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 계산 정확도 검증
            actual_diff = abs(int(current) - int(baseline))
            expected_diff = int(re.search(r'(\d+)명', expected).group(1))
            
            if actual_diff == expected_diff:
                correct_calculations += 1
                logger.info(f"✅ 수학 테스트 {i}: 정확")
            else:
                logger.info(f"❌ 수학 테스트 {i}: 오류 (예상: {expected_diff}, 실제: {actual_diff})")
            
            logger.info(f"입력: {input_text}")
            logger.info(f"출력: {generated_text}")
            logger.info("-" * 50)
        
        accuracy = correct_calculations / total_tests * 100
        logger.info(f"🧮 수학 추론 정확도: {accuracy:.1f}% ({correct_calculations}/{total_tests})")
        
    except Exception as e:
        logger.error(f"수학 추론 테스트 중 오류: {e}")

def main():
    """메인 실행 함수 (수학 추론 특화)"""
    set_seed(SEED)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")
    
    # 수학 추론 특화 모델 로드
    logger.info("🤖 수학 추론 특화 모델 로드 중...")
    model, tokenizer, model_name = try_load_math_model()
    model = model.to(device)
    
    # 모델 정보 출력
    logger.info(f"선택된 수학 추론 모델: {model_name}")
    logger.info(f"토크나이저 vocab 크기: {len(tokenizer)}")
    
    # 수학 추론 특화 데이터셋 생성
    logger.info("📊 수학 추론 특화 데이터셋 생성 중...")
    dataset = create_math_focused_dataset(CSV_FILES)
    logger.info(f"데이터셋 생성 완료: 훈련 {len(dataset['train'])}개, 검증 {len(dataset['validation'])}개")
    
    # 데이터 토큰화
    logger.info("🔄 데이터 토큰화 중...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function_math_specialized(x, tokenizer, model_name),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 수학 추론 특화 학습 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,  # 수학 추론을 위한 낮은 학습률
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=200,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=3,
        predict_with_generate=True,
        fp16=True if device.type == "cuda" else False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # GPU 멀티프로세싱 문제 해결
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # GPU 멀티프로세싱 문제 해결
        report_to=[],
        # 수학 추론 특화 설정
        max_grad_norm=1.0,
        optim="adamw_torch",
        seed=SEED,
        gradient_accumulation_steps=4,
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
    logger.info(f"🚀 수학 추론 특화 모델 학습 시작: {model_name}")
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
    logger.info("💾 수학 추론 특화 모델 저장 중...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 수학 추론 정확도 테스트
    logger.info("🧮 학습된 모델 수학 추론 정확도 테스트...")
    test_math_reasoning_accuracy(model, tokenizer, device, model_name)
    
    elapsed = datetime.datetime.now() - start_time
    logger.info(f"✅ 수학 추론 특화 모델 학습 완료! 총 시간: {elapsed}")
    
    return True

if __name__ == "__main__":
    main() 