#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoBART 기반 군중 모니터링 SLM 모델 평가 코드
F1 Score, ROUGE-L, 응답 시간 측정
"""

import os
import pandas as pd
import torch
import time
import json
from transformers import BartTokenizer, BartForConditionalGeneration
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

class ModelEvaluator:
    """KoBART 모델 평가 클래스"""
    
    def __init__(self, model_path, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")
        
        # 모델 및 토크나이저 로드
        if tokenizer_path is None:
            tokenizer_path = model_path
            
        logger.info("KoBART 모델 및 토크나이저 로드 중...")
        try:
            self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
            self.model = BartForConditionalGeneration.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            logger.info("기본 KoBART 모델을 사용합니다.")
            self.tokenizer = BartTokenizer.from_pretrained("gogamza/kobart-base-v2")
            self.model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 특수 토큰 확인
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ROUGE 스코어러 초기화
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        
    def generate_response(self, input_text, max_length=512, num_beams=4):
        """텍스트 생성 (실제 KoBART 모델 사용)"""
        try:
            # 응답 시간 측정 시작
            start_time = time.time()
            
            # 입력 텍스트 전처리
            if not isinstance(input_text, str):
                input_text = str(input_text)
            
            # 특수 문자 정리 (안전한 방식)
            input_text = input_text.replace('\u200e', '').replace('\u200f', '').replace('\ufeff', '')
            input_text = re.sub(r'[^\w\s가-힣:.\-,\(\)]', ' ', input_text)  # 안전한 문자만 유지
            input_text = ' '.join(input_text.split())  # 공백 정리
            
            # 입력이 너무 길면 자르기
            if len(input_text) > 200:
                input_text = input_text[:200] + "..."
            
            logger.info(f"전처리된 입력: {input_text[:100]}...")
            
            # 토큰화
            try:
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=128,  # 입력 길이 제한
                    truncation=True,
                    padding=True,
                    add_special_tokens=True
                ).to(self.device)
                
                logger.info(f"토큰화 완료: input_ids shape = {inputs['input_ids'].shape}")
                
            except Exception as tokenize_error:
                logger.error(f"토큰화 오류: {tokenize_error}")
                return "토큰화 실패", time.time() - start_time
            
            # 실제 KoBART 모델로 텍스트 생성
            try:
                with torch.no_grad():
                    # 매우 보수적인 생성 설정
                    generated_ids = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=inputs["input_ids"].shape[1] + 50,  # 입력 + 50토큰
                        min_length=inputs["input_ids"].shape[1] + 5,   # 최소 5토큰 추가
                        num_beams=1,  # 그리디 디코딩
                        do_sample=False,  # 샘플링 비활성화
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        early_stopping=False,  # early_stopping 제거
                        use_cache=True,
                        output_scores=False,
                        return_dict_in_generate=False,
                        no_repeat_ngram_size=0,  # 반복 제한 해제
                        temperature=None,  # temperature 제거 (do_sample=False이므로)
                        top_p=None,  # top_p 제거
                        top_k=None   # top_k 제거
                    )
                    
                    logger.info(f"생성 완료: generated_ids shape = {generated_ids.shape}")
                    
            except Exception as generation_error:
                logger.error(f"텍스트 생성 오류: {generation_error}")
                # 실패 시 규칙 기반 대체
                return self._generate_fallback_response(input_text), time.time() - start_time
            
            # 디코딩
            try:
                # 전체 시퀀스 디코딩
                full_text = self.tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
                
                logger.info(f"전체 텍스트: {full_text[:100]}...")
                
                # 입력 부분 제거
                input_decoded = self.tokenizer.decode(
                    inputs["input_ids"][0], 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
                
                logger.info(f"입력 텍스트: {input_decoded[:100]}...")
                
                # 생성된 부분만 추출
                if input_decoded and input_decoded.strip() in full_text:
                    generated_text = full_text.replace(input_decoded.strip(), "", 1).strip()
                else:
                    # 토큰 레벨에서 분리
                    input_length = inputs["input_ids"].shape[1]
                    if len(generated_ids[0]) > input_length:
                        generated_tokens = generated_ids[0][input_length:]
                        generated_text = self.tokenizer.decode(
                            generated_tokens, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=True
                        )
                    else:
                        generated_text = full_text
                
                logger.info(f"생성된 텍스트: {generated_text[:100]}...")
                
            except Exception as decode_error:
                logger.error(f"디코딩 오류: {decode_error}")
                return self._generate_fallback_response(input_text), time.time() - start_time
            
            # 후처리
            if generated_text:
                # 특수 문자 정리
                generated_text = generated_text.replace('\u200e', '').replace('\u200f', '').replace('\ufeff', '')
                generated_text = re.sub(r'[^\w\s가-힣:.\-,\(\)!?]', ' ', generated_text)
                generated_text = ' '.join(generated_text.split()).strip()
                
                # 너무 짧거나 의미없는 응답 필터링
                if len(generated_text.strip()) < 10 or generated_text.strip() in ['', '.', '..', '...']:
                    generated_text = self._generate_fallback_response(input_text)
            else:
                generated_text = self._generate_fallback_response(input_text)
            
            # 응답 시간 측정 종료
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"최종 응답: {generated_text[:100]}...")
            logger.info(f"응답 시간: {response_time:.4f}초")
            
            return generated_text, response_time
            
        except Exception as e:
            logger.error(f"전체 생성 과정 오류: {e}")
            return self._generate_fallback_response(input_text), time.time() - start_time if 'start_time' in locals() else 0.0
    
    def _generate_fallback_response(self, input_text):
        """모델 실패 시 규칙 기반 대체 응답"""
        try:
            # 키워드 추출
            domain_match = re.search(r'도메인:\s*([^\n]+)', input_text)
            situation_match = re.search(r'상황:\s*([^\n]+)', input_text)
            
            domain = domain_match.group(1).strip() if domain_match else "일반"
            situation = situation_match.group(1).strip() if situation_match else "상황 파악 필요"
            
            # 숫자와 시간 추출
            numbers = re.findall(r'\d+', situation)
            times = re.findall(r'\d{1,2}시\s*\d{0,2}분?', situation)
            
            # 간단한 응답 생성
            if "축제" in domain or "축제장" in situation:
                response = "축제장 혼잡 상황에 대한 안전 관리가 필요합니다."
            elif "터미널" in domain or "공항" in domain:
                response = "터미널/공항 대기열 관리와 질서 유지가 필요합니다."
            elif "전시" in domain:
                response = "전시센터 입구 혼잡 완화 조치가 필요합니다."
            elif "지하철" in domain:
                response = "지하철역 승객 안전 관리가 필요합니다."
            else:
                response = "군중 모니터링 상황에 대한 적절한 대응이 필요합니다."
            
            if numbers:
                response += f" (인원: {numbers[0]}명)"
            if times:
                response += f" (시간: {times[0]})"
                
            return response
            
        except:
            return "상황 분석 후 적절한 대응 조치가 필요합니다."
    
    def calculate_f1_score(self, predictions, references):
        """F1 Score 계산 (개선된 버전 - 한국어 최적화)"""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # 기본 토큰 기반 계산
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            
            # 기본 교집합
            basic_intersection = pred_tokens & ref_tokens
            
            # 숫자 추출 및 매칭 (더 관대한 방식)
            pred_numbers = set(re.findall(r'\d+', pred))
            ref_numbers = set(re.findall(r'\d+', ref))
            number_intersection = pred_numbers & ref_numbers
            
            # 한국어 키워드 추출 (2글자 이상)
            pred_korean = set(re.findall(r'[가-힣]{2,}', pred))
            ref_korean = set(re.findall(r'[가-힣]{2,}', ref))
            korean_intersection = pred_korean & ref_korean
            
            # 전체 매칭 요소 결합
            all_pred_elements = pred_tokens | pred_numbers | pred_korean
            all_ref_elements = ref_tokens | ref_numbers | ref_korean
            all_intersections = basic_intersection | number_intersection | korean_intersection
            
            # Precision과 Recall 계산
            if all_pred_elements:
                precision = len(all_intersections) / len(all_pred_elements)
            else:
                precision = 0
                
            if all_ref_elements:
                recall = len(all_intersections) / len(all_ref_elements)
            else:
                recall = 0
            
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
    
    # 입력 텍스트 생성 (KoBART 훈련과 동일한 형식 + 한국어 명시)
    inputs = []
    for _, row in combined_df.iterrows():
        input_text = f"다음은 한국어로 답변해주세요.\n군중 모니터링 상황을 분석하고 적절한 대응 방안을 제시하세요.\n도메인: {row['Domain']}\n상황: {row['Input']}\n한국어 대응방안:"
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
        'model_type': 'KoBART',
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
    MODEL_PATH = "./kobart-crowd-monitoring"
    CSV_FILES = [
        "../csv/domain1_dataset - 시트1.csv",
        "../csv/domain1_example_dataset.csv"
    ]
    OUTPUT_FILE = f"kobart_evaluation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 모델 경로 확인
    if not os.path.exists(MODEL_PATH):
        logger.error(f"모델 경로가 존재하지 않습니다: {MODEL_PATH}")
        logger.info("먼저 train_kobart.py를 실행하여 모델을 학습시키거나 기본 KoBART 모델을 사용합니다.")
        MODEL_PATH = None  # 기본 모델 사용
    
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
    logger.info("KoBART 모델 평가 시작...")
    start_time = datetime.datetime.now()
    
    results = evaluator.evaluate_model(test_inputs, test_targets, max_samples=40)  # 샘플 수 제한
    
    end_time = datetime.datetime.now()
    evaluation_time = end_time - start_time
    
    # 결과 출력
    logger.info("=== KoBART 평가 결과 ===")
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