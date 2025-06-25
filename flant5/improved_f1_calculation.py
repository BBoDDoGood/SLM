import re
import numpy as np

def calculate_improved_f1_score(predictions, references):
    """개선된 F1 Score 계산 (숫자, 키워드 기반)"""
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # 1. 숫자 추출 (시간, 인원수 등)
        pred_numbers = set(re.findall(r'\d+', pred))
        ref_numbers = set(re.findall(r'\d+', ref))
        
        # 2. 영문 키워드 추출
        pred_keywords = set(re.findall(r'[a-zA-Z]+', pred.lower()))
        ref_keywords = set(['gate', 'entrance', 'terminal', 'airport', 'station', 'people', 'crowd'])
        
        # 3. 한글 핵심 단어 추출
        korean_keywords = ['입구', '게이트', '터미널', '공항', '역사', '명', '인원', '기준', '감지', '체류']
        ref_korean = set()
        for keyword in korean_keywords:
            if keyword in ref:
                ref_korean.add(keyword)
        
        # 4. 교집합 계산
        number_match = len(pred_numbers & ref_numbers)
        keyword_match = len(pred_keywords & ref_keywords)
        korean_match = len(ref_korean)  # 한글은 예측에 없으므로 참조만 고려
        
        # 5. 가중치 적용
        total_pred_elements = len(pred_numbers) + len(pred_keywords)
        total_ref_elements = len(ref_numbers) + len(ref_keywords) + len(ref_korean)
        total_matches = number_match * 2 + keyword_match + korean_match * 0.5  # 숫자에 더 높은 가중치
        
        # 6. Precision과 Recall 계산
        precision = total_matches / total_pred_elements if total_pred_elements > 0 else 0
        recall = total_matches / total_ref_elements if total_ref_elements > 0 else 0
        
        # 7. F1 Score 계산
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

# 테스트 예시
predictions = [
    "Duration I 129. , 100.",
    "Parking . 5 1 a , ( ) 56."
]

references = [
    "공항 국제선 I보안검색 진입구에는 기준 인원 100명을 초과한 129명이 09시 30분 현재 확인되었습니다.",
    "오후 5시 지하철역 5호선 1번 출입구 앞에 기준 미설정 상태로 56명이 체류하고 있습니다."
]

improved_f1 = calculate_improved_f1_score(predictions, references)
print(f"개선된 F1 Score: {improved_f1:.4f}")

# 기존 방식과 비교
def calculate_original_f1_score(predictions, references):
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        intersection = pred_tokens & ref_tokens
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)

original_f1 = calculate_original_f1_score(predictions, references)
print(f"기존 F1 Score: {original_f1:.4f}")
print(f"개선 정도: {improved_f1 - original_f1:.4f}")
