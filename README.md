# 📊 모델 평가 비교 프로젝트

한국어 Small Language Model (SLM) 성능 비교를 위한 군중 모니터링 도메인 특화 모델 학습 및 평가 프로젝트

## 🎯 프로젝트 개요

이 프로젝트는 **군중 모니터링** 도메인에서 한국어 SLM 모델들의 성능을 비교 평가합니다.

### 📋 평가 대상 모델
- **FlanT5** (`google/flan-t5-base`)
- **KoBART** (`gogamza/kobart-base-v2`)
- **PKO-T5** (`pko-t5-base`)

## 📁 프로젝트 구조

```
모델 평가 비교/
├── flant5/                 # FlanT5 모델 관련 코드
│   ├── train_flant5.py     # 학습 스크립트
│   ├── test_flant5.py      # 평가 스크립트
│   └── *.json              # 평가 결과
├── kobart/                 # KoBART 모델 관련 코드
│   ├── train_kobart.py     # 학습 스크립트
│   ├── test_kobart.py      # 평가 스크립트
│   └── *.json              # 평가 결과
├── pko-t5/                 # PKO-T5 모델 관련 코드
│   ├── train_pko_t5.py     # 학습 스크립트
│   ├── test_pko_t5.py      # 평가 스크립트
│   └── *.json              # 평가 결과
├── csv/                    # 학습/평가 데이터셋
├── SLM_dataset/            # 도메인별 데이터셋
└── requirement.txt         # 의존성 패키지
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
pip install -r requirement.txt
```

### 2. 모델 학습
```bash
# FlanT5 학습
cd flant5
python train_flant5.py

# KoBART 학습
cd kobart
python train_kobart.py

# PKO-T5 학습
cd pko-t5
python train_pko_t5.py
```

### 3. 모델 평가
```bash
# 각 모델 디렉토리에서 평가 실행
python test_[모델명].py
```

## 📊 평가 결과

각 모델별 평가 결과는 JSON 형태로 저장되며, 다음 지표들을 포함합니다:
- **BLEU Score**: 번역 품질 평가
- **ROUGE Score**: 요약 품질 평가
- **F1 Score**: 전반적인 성능 지표
- **Exact Match**: 정확한 일치율

## 🎨 주요 특징

- **한국어 특화**: 한국어 처리에 최적화된 모델들 비교
- **도메인 특화**: 군중 모니터링 도메인에 특화된 학습
- **자동화된 평가**: 일관된 평가 지표로 객관적 비교
- **메모리 최적화**: 제한된 리소스에서도 효율적 학습

## 🔧 기술 스택

- **Python 3.8+**
- **PyTorch**
- **Transformers (Hugging Face)**
- **Pandas**
- **Scikit-learn**

## 📈 성능 비교

| 모델 | BLEU | ROUGE-L | F1 | 학습 시간 |
|------|------|---------|----|---------| 
| FlanT5 | - | - | - | - |
| KoBART | - | - | - | - |
| PKO-T5 | - | - | - | - |

*평가 진행 중*

## 🤝 기여 방법

1. 이 저장소를 Fork
2. 새로운 기능 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경사항 커밋 (`git commit -am '새 기능 추가'`)
4. 브랜치에 Push (`git push origin feature/새기능`)
5. Pull Request 생성

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 있습니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해 주세요.

---

*군중 모니터링을 위한 한국어 SLM 성능 비교 연구* 🚀 