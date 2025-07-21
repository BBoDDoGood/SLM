#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKO-T5 기반 하이브리드 SLM 모델 학습 코드
기존 학습 스크립트에 하이브리드 시스템 통합
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
import sys

# 상위 디렉토리 추가 (하이브리드 모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 하이브리드 모듈 import
try:
    from math_calculator import MathCalculator
    from hybrid_slm_generator import HybridSLMGenerator
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("⚠️ 하이브리드 모듈을 import할 수 없습니다. 기본 모드로 실행됩니다.")

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
        logging.FileHandler(f'pko_t5_hybrid_train_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(csv_files):
    """CSV 파일들에서 데이터셋 로드 (하이브리드 지원)"""
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
        
        # 하이브리드 시스템을 위한 입력 포맷
        df['input_text'] = "분석: " + df['Domain'] + ", " + df['Input']
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
    """안전한 전처리 함수 (하이브리드 지원)"""
    try:
        # 입력 텍스트 처리
        inputs = examples['input_text']
        targets = examples['target_text']
        
        # 토크나이저 설정
        model_inputs = tokenizer(
            inputs,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # 타겟 텍스트 처리
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        
        # 패딩 토큰을 -100으로 설정 (loss 계산에서 제외)
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
        
    except Exception as e:
        logger.error(f"전처리 실패: {e}")
        raise

def try_load_model(model_name):
    """모델 로드 시도"""
    for model in MODEL_OPTIONS:
        try:
            logger.info(f"모델 로드 시도: {model}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model)
            
            # 토크나이저 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"✅ 모델 로드 성공: {model}")
            return model_obj, tokenizer
            
        except Exception as e:
            logger.warning(f"모델 로드 실패: {model}, 오류: {e}")
            continue
    
    raise ValueError("사용 가능한 모델을 찾을 수 없습니다.")

def test_hybrid_system(model_path, test_cases=None):
    """하이브리드 시스템 테스트"""
    if not HYBRID_AVAILABLE:
        logger.warning("하이브리드 모듈을 사용할 수 없습니다.")
        return None
    
    try:
        logger.info("🧪 하이브리드 시스템 테스트 시작")
        
        # 테스트 케이스 준비
        if test_cases is None:
            test_cases = [
                "농구경기장 계단에 12:14 현재 4명이 있으며, 운영 기준은 9명입니다.",
                "13:13 클리닉에서 71명 밀집, 기준 수용인원 49명",
                "8:29 구청 309명 계측, 기준 인원 401명"
            ]
        
        # 하이브리드 생성기 초기화
        generator = HybridSLMGenerator(model_path)
        
        # 테스트 실행
        results = []
        for i, test_input in enumerate(test_cases, 1):
            logger.info(f"테스트 {i}: {test_input}")
            
            result = generator.analyze_generation(test_input)
            results.append(result)
            
            if result["calculation"]:
                logger.info(f"  계산: {result['calculation']['description']}")
                if result["improved"]:
                    logger.info("  ✅ 개선됨")
                elif result["hybrid_correct"]:
                    logger.info("  ✅ 정확함")
                else:
                    logger.warning("  ❌ 여전히 오류")
        
        # 결과 요약
        total = len(results)
        improved = sum(1 for r in results if r["improved"])
        correct = sum(1 for r in results if r["hybrid_correct"])
        
        logger.info(f"📊 하이브리드 테스트 결과:")
        logger.info(f"총 테스트: {total}개")
        logger.info(f"개선된 케이스: {improved}개 ({improved/total*100:.1f}%)")
        logger.info(f"정확한 계산: {correct}개 ({correct/total*100:.1f}%)")
        
        return results
        
    except Exception as e:
        logger.error(f"하이브리드 시스템 테스트 실패: {e}")
        return None

def main():
    """메인 함수 (하이브리드 지원)"""
    try:
        logger.info("🚀 하이브리드 SLM 모델 학습 시작")
        
        # 시드 설정
        set_seed(SEED)
        
        # 데이터셋 로드
        logger.info("📊 데이터셋 로드 중...")
        dataset = load_dataset(CSV_FILES)
        logger.info(f"데이터셋 로드 완료: 훈련 {len(dataset['train'])}개, 검증 {len(dataset['validation'])}개")
        
        # 모델 로드
        logger.info("🤖 모델 로드 중...")
        model, tokenizer = try_load_model(MODEL_OPTIONS[0])
        
        # 전처리 함수 설정
        def preprocess_function(examples):
            return preprocess_function_safe(examples, tokenizer)
        
        # 데이터셋 전처리
        logger.info("🔄 데이터셋 전처리 중...")
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # 학습 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            fp16=False,  # CPU 학습
            dataloader_num_workers=4,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # wandb 비활성화
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # 트레이너 생성
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # 모델 학습
        logger.info("🎯 모델 학습 시작...")
        trainer.train()
        
        # 모델 저장
        logger.info("💾 모델 저장 중...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # 하이브리드 시스템 테스트
        if HYBRID_AVAILABLE:
            logger.info("🧪 하이브리드 시스템 테스트...")
            test_hybrid_system(OUTPUT_DIR)
        else:
            logger.warning("하이브리드 시스템 테스트를 건너뜁니다.")
        
        logger.info("✅ 학습 완료!")
        
    except Exception as e:
        logger.error(f"학습 실패: {e}")
        raise

if __name__ == "__main__":
    main() 