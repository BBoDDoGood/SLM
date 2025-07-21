#!/usr/bin/env python3
"""
PKO-T5 기반 SLM 모델을 Hugging Face Hub에 업로드
"""

import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def upload_model_to_hf():
    # 설정
    MODEL_PATH = "/Volumes/Data/slm_model/checkpoint-2300"  # 최신 체크포인트
    HF_USERNAME = "BBoDDoGood"  # 여기에 허깅페이스 사용자명 입력
    MODEL_NAME = "SLM_pko-t5"  # 원하는 모델명
    REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
    
    print("🚀 PKO-T5 SLM 모델 업로드 시작")
    print(f"모델 경로: {MODEL_PATH}")
    print(f"업로드 대상: {REPO_ID}")
    
    # API 초기화
    api = HfApi()
    
    try:
        # 1. 레포지토리 생성
        print("\n1️⃣ Hugging Face 레포지토리 생성 중...")
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False  # 공개 모델로 설정 (원하면 True로 변경)
        )
        print(f"✅ 레포지토리 생성 완료: https://huggingface.co/{REPO_ID}")
        
        # 2. 모델 파일 업로드
        print("\n2️⃣ 모델 파일 업로드 중...")
        
        # 업로드할 파일들
        files_to_upload = [
            "model.safetensors",      # 메인 모델 (1.0GB)
            "config.json",            # 모델 설정
            "generation_config.json", # 생성 설정
            "tokenizer.json",         # 토크나이저 (4.2MB)
            "tokenizer_config.json",  # 토크나이저 설정
            "special_tokens_map.json" # 특수 토큰 매핑
        ]
        
        for file_name in files_to_upload:
            file_path = os.path.join(MODEL_PATH, file_name)
            if os.path.exists(file_path):
                print(f"  📤 {file_name} 업로드 중...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"  ✅ {file_name} 업로드 완료")
            else:
                print(f"  ⚠️  {file_name} 파일 없음")
        
        # 3. README.md 생성 및 업로드
        print("\n3️⃣ README.md 생성 중...")
        readme_content = f"""---
language:
- ko
license: mit
tags:
- korean
- t5
- crowd-monitoring
- slm
- pko-t5
datasets:
- custom
pipeline_tag: text2text-generation
---

# PKO-T5 SLM: 군중 모니터링 전문 모델

이 모델은 PKO-T5를 기반으로 군중 모니터링 및 상황 분석을 위해 특별히 훈련된 Small Language Model(SLM)입니다.

## 모델 정보

- **Base Model**: `paust/pko-t5-base`
- **Language**: Korean (한국어)
- **Task**: Text-to-Text Generation
- **Domain**: 군중 모니터링, 상황 분석, 안전 관리

## 지원 도메인

1. **군중 밀집 및 체류 감지**
2. **쓰러짐 및 장기 정지 감지**  
3. **연기 및 화염 감지**
4. **작업자 안전장비 미착용 감지**
5. **폐쇄시간 무단 출입 감지**
6. **안전장비 미착용 감지**
7. **온도 기반 쾌적도 및 폭염 예보 안내**
8. **히트맵 기반 체류 위험구간 분석**
9. **줄 서기 및 대기열 정렬 상태 감지**
10. **이상 이동 패턴 감지**

## 사용 방법

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}")
model = AutoModelForSeq2SeqLM.from_pretrained("{REPO_ID}")

# 입력 예시
input_text = "군중 밀집 및 체류 감지, 오후 2시 30분에 지하철 2호선 강남역에서 50명이 5분간 체류했습니다, 기준: 30명"

# 토큰화 및 생성
inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
```

## 훈련 데이터

- **총 데이터 수**: 100,000개 (각 도메인 10,000개)
- **훈련 스텝**: 19,999 steps (99.995% 완료)
- **손실값**: 0.33 (eval_loss)
- **훈련 시간**: 약 1.5시간

## 라이선스

MIT License

## 인용

```bibtex
@model{{pko-t5-slm-crowd-monitoring,
  title={{PKO-T5 SLM: Korean Crowd Monitoring Specialized Model}},
  author={{SLM Development Team}},
  year={{2025}},
  url={{https://huggingface.co/{REPO_ID}}}
}}
```
"""
        
        # README.md 임시 파일 생성
        readme_path = "/tmp/README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # README.md 업로드
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("✅ README.md 업로드 완료")
        
        print("\n🎉 모델 업로드 완료!")
        print(f"📱 모델 페이지: https://huggingface.co/{REPO_ID}")
        print(f"💻 사용법: transformers.AutoModel.from_pretrained('{REPO_ID}')")
        
    except Exception as e:
        print(f"❌ 업로드 중 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    upload_model_to_hf()