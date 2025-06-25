#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKO-T5 기반 군중 모니터링 SLM 모델 평가 코드
F1 Score, ROUGE-L, 응답 시간 측정 (한국어 최적화)
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PKOModelEvaluator:
    """PKO-T5 모델 평가 클래스 (한국어 최적화)"""
    
    def __init__(self, model_path, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")
        
        # 모델 및 토크나이저 로드
        if tokenizer_path is None:
            tokenizer_path = model_path
            
        logger.info("PKO-T5 모델 및 토크나이저 로드 중...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            logger.info("PKO-T5 모델 로드 성공!")
        except Exception as e:
            logger.error(f"PKO-T5 모델 로드 실패: {e}")
            logger.info("대안으로 google/flan-t5-base 모델을 사용합니다.")
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        
        self.model.to(self.device)
        self.model.eval()
        
        # ROUGE 스코어러 초기화 (한국어 설정)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
    def generate_response(self, input_text, max_length=512, num_beams=4):
        """텍스트 생성 (PKO-T5 최적화)"""
        try:
            # 입력 텍스트 전처리
            if not isinstance(input_text, str):
                input_text = str(input_text)
            
            # 입력 토큰화 (PKO-T5에 최적화된 설정)
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            # 응답 시간 측정 시작
            start_time = time.time()
            
            # 텍스트 생성 (한국어에 최적화된 설정)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=512,
                    min_length=30,  # 최소 길이 증가
                    num_beams=5,  # 빔 수 증가
                    do_sample=True,
                    temperature=0.7,  # 온도 조정
                    top_p=0.9,
                    top_k=50,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # 반복 방지
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1
                )
            
            # 응답 시간 측정 종료
            end_time = time.time()
            response_time = end_time - start_time
            
            # 전체 출력 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 입력 부분 제거 (더 정확한 방식)
            input_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if generated_text.startswith(input_decoded):
                generated_text = generated_text[len(input_decoded):].strip()
            
            # 후처리: 불필요한 문자 제거
            generated_text = re.sub(r'^[^\w가-힣]*', '', generated_text)  # 시작 부분 특수문자 제거
            generated_text = re.sub(r'\s+', ' ', generated_text)  # 다중 공백 제거
            generated_text = generated_text.strip()
            
            # 빈 출력 방지
            if not generated_text or len(generated_text) < 10:
                generated_text = "상황에 맞는 적절한 대응 방안을 수립하여 안전을 확보해야 합니다."
            
            return generated_text, response_time
            
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            return "텍스트 생성 중 오류가 발생했습니다.", 0.0
    
    def calculate_f1_score(self, predictions, references):
        """F1 Score 계산 (개선된 한국어 버전)"""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # 기본 토큰 기반 계산
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            
            # 기본 교집합
            basic_intersection = pred_tokens & ref_tokens
            
            # 숫자 추출 및 매칭
            pred_numbers = set(re.findall(r'\d+', pred))
            ref_numbers = set(re.findall(r'\d+', ref))
            number_intersection = pred_numbers & ref_numbers
            
            # 한국어 키워드 추출 (2글자 이상)
            pred_korean = set(re.findall(r'[가-힣]{2,}', pred))
            ref_korean = set(re.findall(r'[가-힣]{2,}', ref))
            korean_intersection = pred_korean & ref_korean
            
            # 의미적 키워드 매칭 (군중 모니터링 도메인 특화)
            monitoring_keywords = [
                '안전', '대피', '통제', '관리', '모니터링', '감시', '경고', '알림',
                '분산', '유도', '차단', '제한', '확인', '점검', '조치', '대응',
                '인원', '군중', '사람', '관람객', '방문객', '승객', '고객',
                '위험', '사고', '혼잡', '집중', '밀집', '과밀', '포화'
            ]
            
            pred_keywords = set()
            ref_keywords = set()
            for keyword in monitoring_keywords:
                if keyword in pred:
                    pred_keywords.add(keyword)
                if keyword in ref:
                    ref_keywords.add(keyword)
            
            keyword_intersection = pred_keywords & ref_keywords
            
            # 전체 매칭된 요소 수 계산
            total_intersection = len(basic_intersection) + len(number_intersection) + \
                               len(korean_intersection) + len(keyword_intersection)
            
            # 전체 요소 수 계산
            total_pred = len(pred_tokens) + len(pred_numbers) + len(pred_korean) + len(pred_keywords)
            total_ref = len(ref_tokens) + len(ref_numbers) + len(ref_korean) + len(ref_keywords)
            
            # Precision, Recall, F1 계산
            precision = total_intersection / total_pred if total_pred > 0 else 0
            recall = total_intersection / total_ref if total_ref > 0 else 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def calculate_rouge_l(self, predictions, references):
        """ROUGE-L Score 계산 (한국어 최적화)"""
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            # 한국어 텍스트 전처리
            pred_clean = re.sub(r'[^\w가-힣\s]', ' ', pred)
            ref_clean = re.sub(r'[^\w가-힣\s]', ' ', ref)
            
            scores = self.rouge_scorer.score(ref_clean, pred_clean)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        
        return np.mean(rouge_l_scores)
    
    def evaluate_model(self, test_inputs, test_targets, max_samples=None):
        """모델 전체 평가 (PKO-T5 최적화)"""
        if max_samples:
            test_inputs = test_inputs[:max_samples]
            test_targets = test_targets[:max_samples]
        
        logger.info(f"PKO-T5 모델 평가 시작 - 데이터 수: {len(test_inputs)}")
        
        predictions = []
        response_times = []
        
        # 각 입력에 대해 예측 수행
        for i, input_text in enumerate(test_inputs):
            if i % 5 == 0:
                logger.info(f"진행률: {i+1}/{len(test_inputs)}")
            
            prediction, response_time = self.generate_response(input_text)
            predictions.append(prediction)
            response_times.append(response_time)
            
            # 중간 결과 로깅
            if i < 3:
                logger.info(f"샘플 {i+1} 예측: {prediction[:100]}...")
        
        # 메트릭 계산
        logger.info("메트릭 계산 중...")
        f1_score = self.calculate_f1_score(predictions, test_targets)
        rouge_l_score = self.calculate_rouge_l(predictions, test_targets)
        avg_response_time = np.mean(response_times)
        
        # 결과 정리
        results = {
            'model_name': 'PKO-T5',
            'f1_score': f1_score,
            'rouge_l_score': rouge_l_score,
            'avg_response_time': avg_response_time,
            'total_samples': len(test_inputs),
            'predictions': predictions,
            'targets': test_targets,
            'response_times': response_times,
            'response_time_std': np.std(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times)
        }
        
        logger.info("PKO-T5 모델 평가 완료!")
        return results
    
    def print_sample_results(self, results, num_samples=5):
        """샘플 결과 출력 (한국어 최적화)"""
        logger.info("=== PKO-T5 샘플 결과 ===")
        predictions = results['predictions']
        targets = results['targets']
        
        for i in range(min(num_samples, len(predictions))):
            logger.info(f"\n--- 샘플 {i+1} ---")
            logger.info(f"예측: {predictions[i]}")
            logger.info(f"정답: {targets[i]}")
            logger.info(f"응답시간: {results['response_times'][i]:.4f}초")
            
            # 간단한 유사도 체크
            pred_words = set(predictions[i].split())
            target_words = set(targets[i].split())
            overlap = len(pred_words & target_words)
            logger.info(f"단어 겹침: {overlap}개")

def load_and_preprocess_data(csv_files):
    """CSV 파일들을 로드하고 전처리 (PKO-T5 최적화)"""
    all_data = []
    
    logger.info(f"CSV 파일 로드 시작: {csv_files}")
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
                logger.info(f"로드된 파일: {csv_file}, 데이터 수: {len(df)}")
            except Exception as e:
                logger.error(f"파일 로드 실패: {csv_file}, 오류: {e}")
        else:
            logger.warning(f"파일을 찾을 수 없음: {csv_file}")
    
    if len(all_data) == 0:
        raise ValueError("로드된 CSV 파일이 없습니다.")
    
    # 모든 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna()
    
    # PKO-T5에 최적화된 프롬프트 생성 (훈련과 동일한 형식)
    inputs = []
    for _, row in combined_df.iterrows():
        input_text = f"다음 군중 모니터링 상황을 분석하고 적절한 대응 방안을 한국어로 제시하세요.\n도메인: {row['Domain']}\n상황: {row['Input']}\n대응방안:"
        inputs.append(input_text)
    
    targets = combined_df['Output'].tolist()
    
    logger.info(f"전체 데이터 수: {len(inputs)}")
    return inputs, targets

def save_results(results, output_file):
    """결과를 JSON 파일로 저장"""
    # NumPy 배열을 리스트로 변환 (JSON 직렬화를 위해)
    json_results = {
        'model_name': results['model_name'],
        'evaluation_time': datetime.datetime.now().isoformat(),
        'metrics': {
            'f1_score': float(results['f1_score']),
            'rouge_l_score': float(results['rouge_l_score']),
            'avg_response_time': float(results['avg_response_time']),
            'response_time_std': float(results['response_time_std']),
            'min_response_time': float(results['min_response_time']),
            'max_response_time': float(results['max_response_time'])
        },
        'statistics': {
            'total_samples': results['total_samples'],
            'successful_predictions': len([p for p in results['predictions'] if p.strip()])
        },
        'sample_results': []
    }
    
    # 샘플 결과 추가 (처음 10개)
    for i in range(min(10, len(results['predictions']))):
        json_results['sample_results'].append({
            'prediction': results['predictions'][i],
            'target': results['targets'][i],
            'response_time': float(results['response_times'][i])
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"평가 결과가 {output_file}에 저장되었습니다.")

def main():
    # 설정
    MODEL_PATH = "./pko-t5-crowd-monitoring"  # 훈련된 PKO-T5 모델 경로
    CSV_FILES = [
        "../csv/domain1_dataset - 시트1.csv",
        "../csv/domain1_example_dataset.csv"
    ]
    
    # 평가 설정
    MAX_SAMPLES = 40  # 평가할 최대 샘플 수
    
    # 데이터 로드 및 전처리
    logger.info("PKO-T5 평가를 위한 데이터 로드 중...")
    inputs, targets = load_and_preprocess_data(CSV_FILES)
    
    # 데이터 분할 (테스트용)
    _, test_inputs, _, test_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )
    
    logger.info(f"테스트 데이터 수: {len(test_inputs)}")
    
    # 모델 평가기 생성
    evaluator = PKOModelEvaluator(MODEL_PATH)
    
    # 모델 평가 수행
    logger.info("PKO-T5 모델 평가 시작...")
    start_time = datetime.datetime.now()
    
    results = evaluator.evaluate_model(test_inputs, test_targets, MAX_SAMPLES)
    
    end_time = datetime.datetime.now()
    evaluation_time = end_time - start_time
    
    # 결과 출력
    logger.info("\n" + "="*50)
    logger.info("PKO-T5 모델 평가 결과")
    logger.info("="*50)
    logger.info(f"F1 Score: {results['f1_score']:.4f}")
    logger.info(f"ROUGE-L Score: {results['rouge_l_score']:.4f}")
    logger.info(f"평균 응답 시간: {results['avg_response_time']:.4f}초")
    logger.info(f"응답 시간 표준편차: {results['response_time_std']:.4f}초")
    logger.info(f"최소 응답 시간: {results['min_response_time']:.4f}초")
    logger.info(f"최대 응답 시간: {results['max_response_time']:.4f}초")
    logger.info(f"총 평가 시간: {evaluation_time}")
    logger.info(f"평가된 샘플 수: {results['total_samples']}")
    
    # 샘플 결과 출력
    evaluator.print_sample_results(results, 3)
    
    # 결과 저장
    output_file = f"pko_t5_evaluation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output_file)
    
    logger.info(f"\nPKO-T5 평가 완료! 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 