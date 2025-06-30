#!/usr/bin/env python3
"""
한국어 T5 모델 테스트 스크립트
MPS 오류 방지 및 안전한 테스트
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정
MODEL_DIR = "./test1"
CSV_PATH = "../csv/domain1_example_dataset.csv"

def load_model_safe():
    """안전한 모델 로드"""
    try:
        # CPU 강제 사용 (MPS 오류 방지)
        device = "cpu"
        logger.info(f"사용 디바이스: {device}")
        
        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        model = model.to(device)
        
        logger.info("✅ 모델 로드 성공!")
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None, None, None

def generate_text_safe(model, tokenizer, device, input_text):
    """안전한 텍스트 생성"""
    try:
        # T5 모델용 prefix 추가
        input_with_prefix = "분석: " + input_text
        
        # 토큰화
        inputs = tokenizer(
            input_with_prefix,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        
        # CPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 생성 (안전한 설정)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=1.0
            )
        
        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        logger.error(f"텍스트 생성 실패: {e}")
        return f"오류: {str(e)}"

def calculate_similarity(text1, text2):
    """간단한 유사도 계산"""
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def test_model():
    """모델 테스트 실행"""
    logger.info("="*80)
    logger.info("🔍 한국어 T5 모델 성능 평가")
    logger.info("="*80)
    
    # 모델 로드
    model, tokenizer, device = load_model_safe()
    if model is None:
        logger.error("모델 로드 실패!")
        return
    
    # 테스트 데이터 로드
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
        logger.info(f"테스트 데이터 수: {len(df)}")
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return
    
    # 샘플 테스트 (처음 10개)
    test_samples = df.head(10)
    total_similarity = 0
    successful_tests = 0
    
    for idx, row in test_samples.iterrows():
        try:
            domain = row['Domain']
            input_text = row['Input']
            expected_output = row['Output']
            
            # 입력 생성 (도메인 + 입력)
            combined_input = f"{domain}, {input_text}"
            
            # 텍스트 생성
            generated_output = generate_text_safe(model, tokenizer, device, combined_input)
            
            # 유사도 계산
            similarity = calculate_similarity(expected_output, generated_output)
            total_similarity += similarity
            successful_tests += 1
            
            # 결과 출력
            print(f"\n📝 샘플 {idx + 1}:")
            print("-" * 70)
            print(f"도메인: {domain}")
            print(f"입력: {input_text}")
            print(f"기대 출력: {expected_output}")
            print(f"예측 출력: {generated_output}")
            print(f"유사도 점수: {similarity:.3f}")
            print(f"길이 - 입력: {len(input_text)}, 기대: {len(expected_output)}, 예측: {len(generated_output)}")
            
        except Exception as e:
            logger.error(f"샘플 {idx + 1} 테스트 실패: {e}")
            continue
    
    # 전체 성능 요약
    if successful_tests > 0:
        avg_similarity = total_similarity / successful_tests
        print(f"\n🎯 전체 성능 요약:")
        print(f"평균 유사도: {avg_similarity:.3f}")
        print(f"성공한 테스트: {successful_tests}/{len(test_samples)}")
        
        # 성능 평가
        if avg_similarity >= 0.7:
            print("🏆 우수한 성능!")
        elif avg_similarity >= 0.5:
            print("✅ 양호한 성능!")
        elif avg_similarity >= 0.3:
            print("⚠️ 보통 성능 - 개선 필요")
        else:
            print("❌ 낮은 성능 - 추가 학습 필요")
    
    # 대화형 테스트
    print(f"\n🗣️ 대화형 테스트 (종료하려면 'quit' 입력)")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\n입력 텍스트: ").strip()
            if user_input.lower() in ['quit', 'exit', '종료']:
                break
            
            if not user_input:
                continue
            
            # 도메인 자동 추가
            full_input = f"군중 밀집 및 체류 감지, {user_input}"
            generated = generate_text_safe(model, tokenizer, device, full_input)
            
            print(f"생성된 분석: {generated}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"오류: {e}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_model()