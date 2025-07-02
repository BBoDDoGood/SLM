# 🚀 PKO-T5 기반 군중 모니터링 Small Language Model (SLM)

## 🎯 프로젝트 개요

이 프로젝트는 **PKO-T5 (Korean T5)** 모델을 기반으로 군중 모니터링과 안전 관리 업무에 특화된 한국어 SLM을 개발합니다. 

실시간 군중 밀집 상황을 분석하고, 안전사고 방지를 위한 구체적인 대응 방안을 제시할 수 있는 AI 모델을 목표로 합니다.

### ✨ 주요 특징

- **🇰🇷 한국어 특화**: PKO-T5 모델 기반 한국어 자연어 처리
- **👥 군중 모니터링**: 실시간 인원 수, 밀집도, 위험도 분석
- **🛡️ 안전 관리**: 상황별 맞춤형 안전 조치 및 대응 방안 제시
- **📊 대용량 데이터**: 50,000개 고품질 한국어 학습 데이터
- **⚡ GPU 최적화**: 제한된 리소스 환경 (RTX 3050 4GB) 최적화
- **🎯 실용성**: 실제 현장에서 활용 가능한 구체적 분석 결과

## 🏆 학습 성과

**✅ 성공적인 모델 학습 완료**

- **모델**: PKO-T5-base (paust/pko-t5-base)
- **학습 데이터**: 50,000개 (도메인별 5,000개 × 10개 도메인)
- **학습 에포크**: 2.1 에포크 (2,100 스텝)
- **성능 개선**: 손실값 5.34 → 0.154 (**97% 개선**)
- **모델 크기**: 1.0GB (safetensors 형식)

## 📁 프로젝트 구조

```
SLM/
├── 🤖 pko-t5/                    # 핵심 학습/테스트 코드
│   ├── train_pko_t5_gpu.py       # GPU 최적화 학습 스크립트
│   ├── train_pko_t5_cpu.py       # CPU 학습 스크립트  
│   └── test_pko_t5.py            # 모델 테스트 및 평가
├── 📚 SLM_dataset/              # 도메인별 설계 문서
└── 📋 requirement.txt            # 의존성 패키지
```

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd SLM

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirement.txt
```

### 2️⃣ 모델 학습

```bash
# GPU 학습 (권장)
cd pko-t5
python train_pko_t5_gpu.py

# CPU 학습 (GPU 미사용시)
python train_pko_t5_cpu.py
```

### 3️⃣ 모델 테스트

```bash
# 학습된 모델 테스트
python test_pko_t5.py
```

## 💡 사용 예시

### 입력 예시
```
군중 밀집 및 체류 감지, 15:30 지하철역 출입구에서 감지 인원 85명, 기준 인원 60명
```

### 출력 예시  
```
15:30 지하철역 출입구에서 기준 인원 60명을 25명 초과한 85명이 밀집되어 있습니다. 
점심시간 승객 유입으로 혼잡이 증가하고 있어 안전사고 위험이 우려됩니다. 
즉각적인 인원 분산 조치와 우회 경로 안내가 필요합니다.
```

## 🗂️ 데이터셋 구성

| 도메인 | 설명 | 데이터 수 | 주요 시나리오 |
|--------|------|-----------|---------------|
| Domain 1 | 군중 밀집 및 체류 감지 | 5,000개 | 지하철역, 쇼핑몰, 공연장 등 |
| Domain 2-10 | 기타 안전 관리 도메인 | 45,000개 | 줄서기 정렬, 비상 상황 등 |
| **전체** | **10개 도메인** | **50,000개** | **종합 군중 모니터링** |

## ⚙️ 시스템 요구사항

### 🖥️ 하드웨어
- **GPU**: NVIDIA RTX 3050 (4GB) 이상 권장
- **메모리**: 16GB RAM 이상
- **저장공간**: 10GB 이상 여유 공간

### 🐍 소프트웨어
- **Python**: 3.8 이상
- **PyTorch**: 2.5.0 이상
- **Transformers**: 4.53.0 이상
- **CUDA**: 11.8 이상 (GPU 사용시)

## 🔧 주요 기술 스택

| 카테고리 | 기술 스택 |
|----------|-----------|
| **🤖 AI Framework** | PyTorch, Hugging Face Transformers |
| **📊 Data Processing** | Pandas, NumPy, Datasets |
| **⚡ Training** | Accelerate, TensorBoard |
| **🎯 Model** | PKO-T5-base (Korean T5) |
| **💾 Storage** | SafeTensors, JSON |

## 📈 성능 지표

### 모델 파일 크기
- **model.safetensors**: 1.0GB
- **config.json**: 2KB  
- **tokenizer.json**: 2.4MB
- **전체 모델**: 약 1.0GB

## 🤝 기여 방법

1. 이 저장소를 Fork 합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/새기능`)
3. 변경사항을 커밋합니다 (`git commit -am '새 기능 추가'`)
4. 브랜치에 Push 합니다 (`git push origin feature/새기능`)
5. Pull Request를 생성합니다
