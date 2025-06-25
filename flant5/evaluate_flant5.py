#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAN-T5 기반 군중 모니터링 SLM 모델 평가 코드
F1 Score, ROUGE-L, 응답 시간 측정
"""

import os
import pandas as pd
import torch
import time
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
import numpy as np
import logging
import datetime
from collections import Counter
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, model_path, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")
        
        # 모델 및 토크나이저 로드
        if tokenizer_path is None:
            tokenizer_path = model_path
            
        logger.info("모델 및 토크나이저 로드 중...")
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # ROUGE 스코어러 초기화
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
    def generate_response(self, input_text, max_length=512, num_beams=4):
        """텍스트 생성"""
        # 입력 토큰화
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # 응답 시간 측정 시작
        start_time = time.time()
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 응답 시간 측정 종료
        end_time = time.time()
        response_time = end_time - start_time
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, response_time
    
    def calculate_f1_score(self, predictions, references):
        """F1 Score 계산 (토큰 기반)"""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # 텍스트를 토큰으로 분할
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            
            # 교집합과 합집합 계산
            intersection = pred_tokens & ref_tokens
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
            recall = len(intersection) / len(ref_tokens) if ref_tokens else 0
            
            # F1 Score 계산
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
                
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def calculate_rouge_l(self, predictions, references):
        """ROUGE-L Score 계산"""
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        
        return np.mean(rouge_l_scores)
    
    def evaluate_model(self, test_inputs, test_targets, max_samples=None):
        """모델 전체 평가"""
        if max_samples:
            test_inputs = test_inputs[:max_samples]
            test_targets = test_targets[:max_samples]
        
        logger.info(f"평가 데이터 수: {len(test_inputs)}")
        
        predictions = []
        response_times = []
        
        # 각 입력에 대해 예측 수행
        for i, input_text in enumerate(test_inputs):
            if i % 10 == 0:
                logger.info(f"진행률: {i+1}/{len(test_inputs)}")
            
            prediction, response_time = self.generate_response(input_text)
            predictions.append(prediction)
            response_times.append(response_time)
        
        # 메트릭 계산
        f1_score = self.calculate_f1_score(predictions, test_targets)
        rouge_l_score = self.calculate_rouge_l(predictions, test_targets)
        avg_response_time = np.mean(response_times)
        
        # 결과 정리
        results = {
            'f1_score': f1_score,
            'rouge_l_score': rouge_l_score,
            'avg_response_time': avg_response_time,
            'total_samples': len(test_inputs),
            'predictions': predictions,
            'targets': test_targets,
            'response_times': response_times
        }
        
        return results
    
    def print_sample_results(self, results, num_samples=5):
        """샘플 결과 출력"""
        logger.info("=== 샘플 결과 ===")
        predictions = results['predictions']
        targets = results['targets']
        
        for i in range(min(num_samples, len(predictions))):
            logger.info(f"\n--- 샘플 {i+1} ---")
            logger.info(f"예측: {predictions[i][:200]}...")
            logger.info(f"정답: {targets[i][:200]}...")
            logger.info(f"응답시간: {results['response_times'][i]:.4f}초")

def load_and_preprocess_data(csv_files):
    """CSV 파일들을 로드하고 전처리"""
    all_data = []
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, encoding='utf-8')
            all_data.append(df)
            logger.info(f"로드된 파일: {csv_file}, 데이터 수: {len(df)}")
    
    # 모든 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 결측값 제거
    combined_df = combined_df.dropna()
    
    # 입력 텍스트 생성 (Domain + Input)
    inputs = []
    for _, row in combined_df.iterrows():
        input_text = f"도메인: {row['Domain']} 입력: {row['Input']}"
        inputs.append(input_text)
    
    targets = combined_df['Output'].tolist()
    
    return inputs, targets

def save_results(results, output_file):
    """결과를 파일로 저장"""
    # 응답 시간 통계
    response_times = results['response_times']
    response_time_stats = {
        'mean': np.mean(response_times),
        'std': np.std(response_times),
        'min': np.min(response_times),
        'max': np.max(response_times),
        'median': np.median(response_times)
    }
    
    # 저장할 결과 정리
    save_data = {
        'evaluation_date': datetime.datetime.now().isoformat(),
        'metrics': {
            'f1_score': float(results['f1_score']),
            'rouge_l_score': float(results['rouge_l_score']),
            'avg_response_time': float(results['avg_response_time']),
            'response_time_stats': response_time_stats
        },
        'total_samples': results['total_samples'],
        'sample_results': []
    }
    
    # 샘플 결과 추가 (처음 10개)
    for i in range(min(10, len(results['predictions']))):
        sample = {
            'prediction': results['predictions'][i],
            'target': results['targets'][i],
            'response_time': results['response_times'][i]
        }
        save_data['sample_results'].append(sample)
    
    # JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"평가 결과가 {output_file}에 저장되었습니다.")

def main():
    # 설정
    MODEL_PATH = "./flant5-crowd-monitoring"
    CSV_FILES = [
        "domain1_dataset - 시트1.csv",
        "domain1_example_dataset.csv"
    ]
    OUTPUT_FILE = f"flant5_evaluation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 모델 경로 확인
    if not os.path.exists(MODEL_PATH):
        logger.error(f"모델 경로가 존재하지 않습니다: {MODEL_PATH}")
        logger.info("먼저 train_flant5_crowd_monitoring.py를 실행하여 모델을 학습시키세요.")
        return
    
    # 데이터 로드
    logger.info("데이터 로드 중...")
    inputs, targets = load_and_preprocess_data(CSV_FILES)
    
    # 테스트 데이터 분할 (원래 학습에서 사용한 것과 동일한 분할)
    _, test_inputs, _, test_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )
    
    logger.info(f"테스트 데이터 수: {len(test_inputs)}")
    
    # 평가기 초기화
    evaluator = ModelEvaluator(MODEL_PATH)
    
    # 모델 평가 수행
    logger.info("모델 평가 시작...")
    start_time = datetime.datetime.now()
    
    results = evaluator.evaluate_model(test_inputs, test_targets, max_samples=50)  # 샘플 수 제한
    
    end_time = datetime.datetime.now()
    evaluation_time = end_time - start_time
    
    # 결과 출력
    logger.info("=== 평가 결과 ===")
    logger.info(f"F1 Score: {results['f1_score']:.4f}")
    logger.info(f"ROUGE-L Score: {results['rouge_l_score']:.4f}")
    logger.info(f"평균 응답 시간: {results['avg_response_time']:.4f}초")
    logger.info(f"총 평가 시간: {evaluation_time}")
    logger.info(f"평가 샘플 수: {results['total_samples']}")
    
    # 샘플 결과 출력
    evaluator.print_sample_results(results)
    
    # 결과 저장
    save_results(results, OUTPUT_FILE)
    
    # 상세 통계 출력
    response_times = results['response_times']
    logger.info("\n=== 응답 시간 통계 ===")
    logger.info(f"평균: {np.mean(response_times):.4f}초")
    logger.info(f"표준편차: {np.std(response_times):.4f}초")
    logger.info(f"최소: {np.min(response_times):.4f}초")
    logger.info(f"최대: {np.max(response_times):.4f}초")
    logger.info(f"중간값: {np.median(response_times):.4f}초")

if __name__ == "__main__":
    main() 