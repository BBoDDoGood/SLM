# 🧠 SLM 대표예시 설계 지침서 (도메인별 공통)

> 이 문서는 SLM(Small Language Model)을 활용한 상황 감지 안내 시스템 개발을 위해, 대표예시 데이터를 작성할 때 반드시 지켜야 할 전사 통합 기준입니다. 모든 도메인 작성자 및 자동조합 설계자는 이 기준을 100% 준수해야 합니다.

---

## ✅ 1. Input 작성 기준

### 📌 필수 요소

* 시간
* 장소(층수, 구역명 등)
* 감지 객체 수치 (예: 사람 28명)
* 기준 수치 (있는 경우)

### 📌 구성 방식

* 자연어 흐름 유지 (단어 나열 금지)
* 콤마(,)는 기준 앞에서만 자연스러운 흐름일 경우에만 사용 허용
* 판단/감정 표현 금지 (예: 혼잡, 위험, 붐빔 등은 Output 전용)

### 📌 표현 유형

* 구조형 예시: `14:22 5층 A구역 사람 28명, 기준 인원 20명`
* 설명형 예시: `5층 A구역에 28명이 모여 있으며, 기준 인원은 20명입니다.`

### 📌 다양화 전략

* 구성 요소 순서를 다양한 패턴으로 설계 (예: 장소→수치→시간 등)
* Input 수치 범위: 1명 \~ 1,000명까지 포함
* 기준과 측정값 격차: ±1명, ±5명, 2\~5배 초과 등 폭넓게 구성

---

## ✅ 2. Output 작성 기준

### 📌 기본 구성

* 판단 결과 문장 (예: 초과, 미달, 근접 등)
* 조치 안내 문장 (예: 분산 유도, 확인 요청 등)
* 2\~4문장 사이로 구성

### 📌 표현 다양화

* 동의어 분산 사용: “혼잡합니다” → “정체 발생”, “유입 집중”, “밀집된 상태” 등
* 기준 수치는 항상 언급 (있는 경우)
* Input 장소명은 Output에 동일하게 포함
* 기준 없음일 경우: 상대 비교, 구조 위험, 시간 흐름 등으로 판단 설명

### 📌 시간 표현 처리 기준

* Input의 시간(예: 14:22)은 Output에 그대로 복사할 필요는 없음
* 단, Output 문장에는 반드시 해당 시간의 의미(현재 시각, 이 시점, 시간 기준 등)를 포함해야 함
* 시간 수치가 완전히 누락되는 문장은 금지

---

## ✅ 3. Input–Output 일관성 원칙

### 📌 반드시 지켜야 할 사항

* Output에 등장하는 수치, 장소, 시간, 기준은 Input과 정확히 일치해야 함
* Output에서 Input에 없는 정보(예: 다른 층, 잘못된 수치, 안내 인력 출동 등) 생성 금지
* Output에 “기준보다 8명 초과”와 같은 계산형 표현은 허용 (단 계산 정확해야 함)

---

## ✅ 4. 대표예시 데이터 비율 기준 (100개 기준)

| 항목        | 기준 비율   |
| --------- | ------- |
| 혼잡/위험 상황  | 70%     |
| 예보/주의 상황  | 20%     |
| 정상/안정 상황  | 10%     |
| 기준 있음     | 60\~70% |
| 기준 없음     | 30\~40% |
| 구역명 포함    | 60\~70% |
| 구역명 생략    | 30\~40% |
| 구조형 Input | 약 60%   |
| 설명형 Input | 약 40%   |

---

## ✅ 5. 표현 방식 다양화 지침

* 혼잡, 붐빔, 정체, 집중, 몰림, 유입 증가 등 유의어 다양하게 활용
* 조치 문장도 “안내 바랍니다”, “유도해 주세요”, “현장 확인 요망”, “우회 권장” 등 분산 작성
* 기준 없음 표현도 반복 없이 분산 사용 (예: 기준 없음 / 기준값 없음 / 미설정 상태 등)

---

## ✅ 6. Input–Output 예시 템플릿

**Input 예시:**

```
① 14:22 백화점 5층 A구역 사람 28명, 기준 인원 20명
② 백화점 5층 A구역 기준 인원 20명, 현재 인원 28명, 시각 14:22
③ 현재 시각은 14시 22분이며, A구역에 28명이 있습니다. 기준은 20명입니다.
```

**Output 예시:**

```
5층 A구역에 기준 인원 20명을 초과한 28명이 체류 중입니다.  
정체가 발생하고 있어 우회 안내가 필요합니다.
```

---

## ✅ 7. 지침 사용 방법

* 대표예시 수작업 작성자: 지침 1\~6 모두 따라야 함
* 자동조합 설계자: 모든 비율 기준, 다양화 전략 적용 필수
* 검수자: Output 수치·장소·판단 기준이 Input과 일치하는지 반드시 확인

---

## ✅ 최종 정리

대표예시는 단순한 예시 문장이 아니라,
SLM이 판단 흐름과 말투를 정확히 익히도록 설계된 “학습 기준”입니다.
지침을 100% 준수해 작성된 데이터만이 안정적이고 응용력 있는 언어모델을 만들 수 있습니다.
