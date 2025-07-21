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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_PATH = "/Users/yunseong/Desktop/SLM_Model/csv/domain_example.csv"

def load_model_safe():
    try:
        device = "cpu"
        logger.info(f"사용 디바이스: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained("BBoDDoGood/SLM_pko-t5", use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained("BBoDDoGood/SLM_pko-t5")
        model = model.to(device)
        
        logger.info("[SUCCESS] 모델 로드 성공!")
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None, None, None

def generate_text_safe(model, tokenizer, device, input_text):
    try:
        input_with_prefix = "분석: " + input_text
        
        inputs = tokenizer(
            input_with_prefix,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        logger.error(f"텍스트 생성 실패: {e}")
        return f"오류: {str(e)}"



def test_model():
    """모델 테스트 실행"""
    logger.info("="*80)
    logger.info("[TEST] 한국어 T5 모델 성능 평가")
    logger.info("="*80)
    
    model, tokenizer, device = load_model_safe()
    if model is None:
        logger.error("모델 로드 실패!")
        return
    
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
        logger.info(f"테스트 데이터 수: {len(df)}")
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return
    
    test_samples = df.head(10)
    successful_tests = 0
    
    for idx, row in test_samples.iterrows():
        try:
            domain = row['Domain']
            input_text = row['Input']
            
            combined_input = f"{domain}, {input_text}"
            
            generated_output = generate_text_safe(model, tokenizer, device, combined_input)
            
            successful_tests += 1
            
            print(f"\n[SAMPLE] 샘플 {successful_tests}:")
            print("-" * 70)
            print(f"도메인: {domain}")
            print(f"입력: {input_text}")
            print(f"예측 출력: {generated_output}")
            
        except Exception as e:
            logger.error(f"샘플 {successful_tests + 1} 테스트 실패: {e}")
            continue
    
    if successful_tests > 0:
        print(f"\n[SUMMARY] 전체 성능 요약:")
        print(f"성공한 테스트: {successful_tests}/{len(test_samples)}")
        print("[COMPLETE] 테스트 완료")
    
    # 대화형 테스트
    print(f"\n[INTERACTIVE] 대화형 테스트 (종료하려면 'quit' 입력)")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\n입력 텍스트: ").strip()
            if user_input.lower() in ['quit', 'exit', '종료']:
                break
            
            if not user_input:
                continue
            
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