import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain3SmokeFlameDetectionGenerator:
    """
    도메인3 연기 및 화염 감지 데이터셋 생성기
    
    주요 기능:
    - 다양한 장소의 연기/화염 감지 상황 데이터 생성
    - 감지 시간과 기준 시간 비교 분석
    - 위험도별 적절한 조치 방안 제시
    - 단일/복합 감지 유형 처리
    - 자연스러운 2~3문장 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 감지 시간 범위 (초)
        self.detection_time_ranges = {
            "단기": {"min": 2, "max": 5, "weight": 40},      # 2-5초, 40%
            "중기": {"min": 6, "max": 10, "weight": 35},     # 6-10초, 35%
            "장기": {"min": 11, "max": 20, "weight": 20},    # 11-20초, 20%
            "초장기": {"min": 21, "max": 60, "weight": 5}    # 21-60초, 5%
        }
        
        # 기준 시간 범위 (초)
        self.baseline_time_ranges = {
            "엄격": {"min": 1, "max": 3, "weight": 30},      # 1-3초, 30%
            "보통": {"min": 4, "max": 7, "weight": 45},      # 4-7초, 45%
            "관대": {"min": 8, "max": 15, "weight": 25}      # 8-15초, 25%
        }
        
        # 감지 유형
        self.detection_types = {
            "연기": {"weight": 35, "expressions": ["연기", "연기가", "연기·불꽃", "연기·화염"]},
            "화염": {"weight": 25, "expressions": ["화염", "화염이", "화염·연기", "화염·불꽃"]},
            "불꽃": {"weight": 20, "expressions": ["불꽃", "불꽃이", "불꽃·연기"]},
            "복합": {"weight": 20, "expressions": ["연기·화염 동시", "불꽃·연기 동시", "화염·연기 동시", "연기와 불꽃", "화염과 연기"]}
        }
        
        # 장소명 - 연기/화염이 감지될 수 있는 다양한 장소
        self.locations = [
            # 산업시설
            "제철소", "화학공장", "자동차공장", "조선소", "석유화학공장", "공장", "정유공장",
            "물류센터", "물류창고", "제조공장",
            
            # 상업시설
            "쇼핑몰", "백화점", "대형마트", "마트", "편의점", "상가", "카페", "레스토랑",
            "미용실", "노래방", "세탁소", "드라이브스루",
            
            # 의료시설
            "병원", "종합병원", "동물병원", "치과", "약국", "요양원", "양로원",
            
            # 교육시설
            "학교", "대학교", "대학", "연구소",
            
            # 교통시설
            "지하철역", "공항", "터미널", "고속도로 휴게소",
            
            # 숙박/여가시설
            "호텔", "펜션", "모텔", "리조트", "찜질방", "사우나", "스포츠센터",
            "체육관", "수영장", "영화관", "공연장", "놀이공원",
            
            # 주거시설
            "아파트", "빌딩",
            
            # 기타시설
            "주유소", "농장", "웨딩홀", "도서관", "공원"
        ]
        
        # 장소별 전용 구역명 매핑
        self.location_specific_areas = {
            # 산업시설
            "제철소": ["3호동 용광로 앞", "용광로", "보일러실"],
            "화학공장": ["반응기 1호", "화학저장고"],
            "자동차공장": ["도장라인", "도장부스"],
            "공장": ["2라인 보일러실", "용접작업장", "남동쪽 용접장", "포장라인", "기계실", 
                   "창고", "남동쪽 적재장", "전기실", "냉동창고"],
            "정유공장": ["저장탱크"],
            "제조공장": ["품질검사실"],
            "물류센터": ["A동 창고", "남동쪽 적재장"],
            "물류창고": ["적재구역"],
            
            # 상업시설
            "쇼핑몰": ["지하1층 푸드코트", "의류매장"],
            "백화점": ["식품매장", "화장품 매장"],
            "마트": ["계산대", "냉동식품 코너"],
            "상가": ["B동 카페 테라스"],
            "카페": ["테라스", "홀", "야외석"],
            "레스토랑": ["주방 조리대"],
            "미용실": ["파마존"],
            "노래방": ["3호실", "개인실"],
            
            # 의료시설
            "병원": ["A동 응급실 입구", "응급실", "중환자실"],
            "종합병원": ["산부인과"],
            
            # 교육시설
            "학교": ["북관 과학실", "화학실험실", "급식실", "운동장"],
            "대학교": ["실습실", "강의실"],
            "대학": ["도서관 열람실"],
            "연구소": ["실험실 B동"],
            
            # 교통시설
            "지하철역": ["2호선 승강장", "환승통로"],
            "공항": ["국제선 대기실", "면세점", "수하물 처리장"],
            "고속도로 휴게소": ["음식점"],
            
            # 숙박/여가시설
            "호텔": ["1층 로비", "연회장", "레스토랑 주방"],
            "펜션": ["A동 복도", "바베큐장", "객실", "객실 발코니"],
            "모텔": ["복도", "엘리베이터홀"],
            "리조트": ["스파 사우나실"],
            "찜질방": ["휴게실", "사우나실"],
            "스포츠센터": ["수영장 탈의실"],
            "체육관": ["동관 농구코트", "농구코트"],
            "공연장": ["무대 뒤편", "객석", "무대 조명실"],
            "놀이공원": ["회전목마"],
            
            # 주거시설
            "아파트": ["7동 13층 복도", "옥상", "지하 주차장", "옥상 기계실", "관리사무소"],
            "빌딩": ["지하주차장"],
            
            # 기타시설
            "농장": ["북쪽 창고", "축사"],
            "웨딩홀": ["신부대기실"],
            "도서관": ["열람실"],
            "터미널": ["대합실", "승차장"]
        }
        
        # 상황 유형별 패턴
        self.situation_patterns = {
            "극도위험": {  # 기준의 3배 이상 초과
                "weight": 15,
                "exceed_ratio": (3.0, 10.0),
                "keywords": ["극도로 위험한", "극도위험", "폭발 위험"]
            },
            "위험상황": {  # 기준의 1.5배~3배 초과
                "weight": 35,
                "exceed_ratio": (1.5, 3.0),
                "keywords": ["위험 수준", "위험상황", "위험 상황"]
            },
            "초과상황": {  # 기준의 1.1배~1.5배 초과
                "weight": 30,
                "exceed_ratio": (1.1, 1.5),
                "keywords": ["초과", "넘어", "넘겨"]
            },
            "정상상황": {  # 기준 이내
                "weight": 15,
                "exceed_ratio": (0.3, 0.9),
                "keywords": ["정상 범위", "안전한", "정상"]
            },
            "기준없음": {  # 기준 미설정
                "weight": 5,
                "exceed_ratio": None,
                "keywords": ["기준 없음", "미설정", "설정되지"]
            }
        }
        
        # 시간 표현 형식
        self.time_formats = [
            "HH:MM",           # 08:12, 14:33
            "새벽 H시 M분",     # 새벽 4시 12분
            "오전 H시 M분",     # 오전 11시 16분
            "오후 H시 M분",     # 오후 7시 18분
            "저녁 H시 M분",     # 저녁 9시 15분
            "HH시 MM분"        # 16시 29분
        ]
        
        # 조치 표현
        self.action_expressions = {
            "극도위험": [
                "즉시 소방대 특수 장비 출동을 요청하고, 공장 전체 작업자 긴급 대피를 실시하세요",
                "즉시 소방대 특수 진압팀 출동과 주변 주민 대피를 요청하세요",
                "소방대 긴급 출동과 전기 전문팀 지원을 요청하세요",
                "즉시 건물 전체 대피를 검토하세요",
            ],
            "위험상황": [
                "즉시 운영을 중단하고 안전 대피를 실시하십시오",
                "즉시 해당 구역 운영을 중단하고 소방팀 출동을 요청하세요",
                "즉시 현장 확인하고 해당 구역 출입을 제한하십시오",
                "소방팀 출동 요청 후 추가 감지 시 전체 대피를 검토하세요",
                "즉시 운영 중단하고 고객 안전 대피를 실시하세요"
            ],
            "초과상황": [
                "해당 구역 점검 후 추가 감지 시 운영 중단을 고려하십시오",
                "즉시 현장을 점검하고 안전 확인을 실시하십시오",
                "해당 구역 이용을 일시 중단하고 안전 점검을 실시하세요",
                "추가 감지되면 해당 구역 대피를 검토하세요",
                "즉시 점검하고 추가 감지 시 소방 안전 점검을 요청하십시오"
            ],
            "정상상황": [
                "일반 점검만 하시면 되며, 추후 감지 시 안전 점검을 실시하세요",
                "상태만 확인하시면 되며, 이후 감지 시 전기 안전 점검을 실시하시기 권합니다",
                "정상 작업을 지속하십시오",
                "점검하고 정상 이용에 지장이 없도록 하십시오",
                "확인하고, 향후 감지 시 종합 안전 점검을 시행하십시오"
            ],
            "기준없음": [
                "즉시 위험 판단은 어려우나 안전 점검을 실시해 주세요",
                "기준이 설정되지 않았지만 안전을 위해 점검하십시오",
                "즉시 위험하지는 않지만 시설 점검을 실시하십시오",
                "임계값이 설정되지 않았으나 점검해 주세요",
                "기준 없이 상황 확인이 필요합니다"
            ]
        }
        
        # 기준 표현
        self.baseline_expressions = [
            "기준", "임계값", "허용치", "임계치", "허용치", "기준값"
        ]
        
        # 기준 상태 표현
        self.baseline_status_expressions = [
            "기준 없음", "허용치 없음", "임계값 없음", "기준 미설정", 
            "미설정", "허용치 미설정", "임계값 미설정", "기준값 미설정"
        ]
        
    def generate_detection_time(self) -> int:
        """감지 시간 생성 (초)"""
        weights = [self.detection_time_ranges[key]["weight"] for key in self.detection_time_ranges.keys()]
        selected_range = random.choices(list(self.detection_time_ranges.keys()), weights=weights)[0]
        range_info = self.detection_time_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_baseline_time(self, detection_time: int, situation_type: str) -> int:
        """기준 시간 생성 (초)"""
        if situation_type == "기준없음":
            return None
        elif situation_type == "극도위험":
            # 기준의 3배~10배 초과하도록 기준 설정
            exceed_min, exceed_max = self.situation_patterns["극도위험"]["exceed_ratio"]
            baseline = int(detection_time / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
        elif situation_type == "위험상황":
            # 기준의 1.5배~3배 초과하도록 기준 설정
            exceed_min, exceed_max = self.situation_patterns["위험상황"]["exceed_ratio"]
            baseline = int(detection_time / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
        elif situation_type == "초과상황":
            # 기준의 1.1배~1.5배 초과하도록 기준 설정
            exceed_min, exceed_max = self.situation_patterns["초과상황"]["exceed_ratio"]
            baseline = int(detection_time / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
        else:  # 정상상황
            # 기준의 30%~90% 수준으로 현재 시간 설정
            exceed_min, exceed_max = self.situation_patterns["정상상황"]["exceed_ratio"]
            baseline = int(detection_time / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
    
    def select_situation_type(self) -> str:
        """상황 유형 선택"""
        weights = [self.situation_patterns[key]["weight"] for key in self.situation_patterns.keys()]
        return random.choices(list(self.situation_patterns.keys()), weights=weights)[0]
    
    def generate_detection_type(self) -> str:
        """감지 유형 생성"""
        weights = [self.detection_types[key]["weight"] for key in self.detection_types.keys()]
        selected_type = random.choices(list(self.detection_types.keys()), weights=weights)[0]
        return random.choice(self.detection_types[selected_type]["expressions"])
    
    def generate_time_expression(self) -> str:
        """시간 표현 생성"""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        if hour < 6:
            return f"새벽 {hour}시 {minute}분"
        elif hour < 12:
            if random.random() < 0.3:
                return f"{hour:02d}:{minute:02d}"
            else:
                return f"오전 {hour}시 {minute}분"
        elif hour < 18:
            if random.random() < 0.3:
                return f"{hour:02d}:{minute:02d}"
            else:
                return f"오후 {hour-12 if hour > 12 else hour}시 {minute}분"
        else:
            if random.random() < 0.3:
                return f"{hour:02d}:{minute:02d}"
            else:
                return f"저녁 {hour-12}시 {minute}분"
    
    def generate_location_with_area(self) -> str:
        """장소 + 구역명 생성"""
        location = random.choice(self.locations)
        
        # 80% 확률로 구역명 추가
        if random.random() < 0.8 and location in self.location_specific_areas:
            area = random.choice(self.location_specific_areas[location])
            return f"{location} {area}"
        else:
            return location
    
    def generate_baseline_expression(self, baseline_time: int) -> str:
        """기준시간 표현 생성"""
        if baseline_time is None:
            return random.choice(self.baseline_status_expressions)
        else:
            expr = random.choice(self.baseline_expressions)
            return f"{expr} {baseline_time}초"
    
    def generate_input_data(self) -> Dict:
        """Input 데이터 생성"""
        situation_type = self.select_situation_type()
        detection_time = self.generate_detection_time()
        baseline_time = self.generate_baseline_time(detection_time, situation_type)
        time = self.generate_time_expression()
        location = self.generate_location_with_area()
        detection_type = self.generate_detection_type()
        
        return {
            "situation_type": situation_type,
            "detection_time": detection_time,
            "baseline_time": baseline_time,
            "time": time,
            "location": location,
            "detection_type": detection_type
        }
    
    def generate_input_string(self, data: Dict) -> str:
        """Input 문자열 생성"""
        time = data["time"]
        location = data["location"]
        detection_type = data["detection_type"]
        detection_time = data["detection_time"]
        baseline_str = self.generate_baseline_expression(data["baseline_time"])
        
        # 다양한 Input 형식 생성
        formats = [
            f'"{time} {location}, {detection_type} 감지 {detection_time}초, {baseline_str}"',
            f'"{detection_type} 출현 {detection_time}초 {location}, {baseline_str}, {time}"',
            f'"{time} {location}에서 {detection_type}이 {detection_time}초간 포착되었습니다. {baseline_str}입니다."',
            f'"{detection_type} {detection_time}초 {location} {time}, {baseline_str}"',
            f'"{location}에서 {detection_type}가 {detection_time}초간 관찰되었으며, {time}, {baseline_str}입니다."'
        ]
        
        return random.choice(formats)
    
    def generate_situation_analysis(self, data: Dict) -> str:
        """상황 분석 문장 생성"""
        detection_time = data["detection_time"]
        baseline_time = data["baseline_time"]
        detection_type = data["detection_type"]
        situation_type = data["situation_type"]
        location = data["location"]
        time = data["time"]
        
        if situation_type == "기준없음":
            return f"{location}에서 {time} {detection_type}가 {detection_time}초간 포착된 상황입니다"
        
        elif situation_type in ["극도위험", "위험상황", "초과상황"]:
            diff = detection_time - baseline_time
            baseline_expr = self.generate_baseline_expression(baseline_time).split()[0]  # "기준"만 추출
            
            if situation_type == "극도위험":
                return f"{location}에서 {time} {detection_type}가 {detection_time}초간 발생하여 {baseline_expr} {baseline_time}초를 {diff}초 초과한 극도로 위험한 상황입니다"
            elif situation_type == "위험상황":
                return f"{location}에서 {time} {detection_type}가 {detection_time}초간 지속되어 {baseline_expr} {baseline_time}초를 {diff}초 초과한 위험 상황입니다"
            else:  # 초과상황
                return f"{location}에서 {time} {detection_type}가 {detection_time}초간 감지되어 {baseline_expr} {baseline_time}초를 {diff}초 초과했습니다"
        
        else:  # 정상상황
            baseline_expr = self.generate_baseline_expression(baseline_time).split()[0]
            return f"{location}에서 {time} {detection_type}가 {detection_time}초간 관찰되어 {baseline_expr} {baseline_time}초보다 {baseline_time - detection_time}초 적은 정상 범위입니다"
    
    def generate_immediate_action(self, data: Dict) -> str:
        """즉시 조치 문장 생성"""
        situation_type = data["situation_type"]
        location = data["location"]
        detection_type = data["detection_type"]
        
        # 상황별 기본 조치
        if situation_type == "극도위험":
            base_action = random.choice(self.action_expressions["극도위험"])
        elif situation_type == "위험상황":
            base_action = random.choice(self.action_expressions["위험상황"])
        elif situation_type == "초과상황":
            base_action = random.choice(self.action_expressions["초과상황"])
        elif situation_type == "정상상황":
            base_action = random.choice(self.action_expressions["정상상황"])
        else:  # 기준없음
            base_action = random.choice(self.action_expressions["기준없음"])
        
        # 장소별 담당자 추가
        if "공장" in location or "제철소" in location or "화학" in location:
            staff = random.choice(["야간 관리자가", "현장 관리자가", "작업자가"])
        elif "병원" in location or "의료" in location:
            staff = random.choice(["의료진이", "병원 직원이"])
        elif "학교" in location or "대학" in location:
            staff = random.choice(["담당 교사가", "관리직원이", "담당 교수가"])
        elif "호텔" in location or "펜션" in location:
            staff = random.choice(["호텔 직원이", "펜션 관리자가", "모텔 관리자가"])
        else:
            staff = random.choice(["직원이", "관리자가", "담당자가"])
        
        return f"{staff} {base_action}"
    
    def generate_additional_condition(self, data: Dict) -> str:
        """추가 조건 문장 생성"""
        detection_time = data["detection_time"]
        situation_type = data["situation_type"]
        
        # 추가 감지 시 조치
        if situation_type in ["극도위험", "위험상황"]:
            threshold = detection_time + random.randint(3, 8)
            actions = [
                f"추가 {threshold}초 이상 감지되면 전체 대피를 검토하세요",
                f"{threshold}초 이상 추가 감지 시 소방대 긴급 출동을 요청하세요",
                f"추가 {threshold}초 이상 감지되면 건물 전체 화재 경보 발령을 검토하세요"
            ]
        elif situation_type == "초과상황":
            threshold = detection_time + random.randint(2, 5)
            actions = [
                f"추가 {threshold}초 이상 감지 시 안전 점검을 요청하십시오",
                f"향후 {threshold}초 이상 재발하면 운영을 일시 중단하시기 권합니다",
                f"동일 위치에서 {threshold}초 이상 지속되면 사용을 일시 중단하세요"
            ]
        elif situation_type == "정상상황":
            threshold = detection_time + random.randint(1, 3)
            actions = [
                f"이후 {threshold}초 이상 감지 시 전기 안전 점검을 실시하세요",
                f"향후 {threshold}초 이상 감지될 경우 종합 안전 점검을 진행하세요",
                f"동일 구역에서 {threshold}초 이상 재발 시 사용을 일시 중단하십시오"
            ]
        else:  # 기준없음
            threshold = detection_time + random.randint(2, 6)
            actions = [
                f"동일 위치에서 {threshold}초 이상 재발 시 이용을 일시 중단하시기 권합니다",
                f"향후 {threshold}초 이상 지속되면 안전 점검을 요청하십시오",
                f"동일한 현상이 {threshold}초 이상 지속되면 해당 구역 일시 통제를 검토하세요"
            ]
        
        return random.choice(actions)
    
    def generate_output_content(self, data: Dict) -> str:
        """Output 내용 생성"""
        situation_analysis = self.generate_situation_analysis(data)
        immediate_action = self.generate_immediate_action(data)
        additional_condition = self.generate_additional_condition(data)
        
        # 2문장 (40%) 또는 3문장 (60%)
        if random.random() < 0.4:
            # 2문장: 상황분석 + 조치
            return f"{situation_analysis}. {immediate_action}."
        else:
            # 3문장: 상황분석 + 즉시조치 + 추가조건
            return f"{situation_analysis}. {immediate_action}. {additional_condition}."
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        input_data = self.generate_input_data()
        input_str = self.generate_input_string(input_data)
        output_str = self.generate_output_content(input_data)
        domain = "연기 및 화염 감지"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인3 데이터셋 {count}개 생성 중...")
        
        for i in range(count):
            if (i + 1) % 100 == 0:
                print(f"   진행률: {i + 1}/{count} ({((i + 1)/count)*100:.1f}%)")
            
            data = self.generate_single_data()
            dataset.append(data)
        
        print(f"✅ 데이터셋 생성 완료: {count}개")
        return dataset
    
    def save_to_csv(self, dataset: List[Tuple[str, str, str]], filepath: str):
        """CSV 파일 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['Input', 'Output', 'Domain'])
            for input_str, output_str, domain in dataset:
                writer.writerow([input_str, output_str, domain])
        
        print(f"💾 CSV 파일 저장 완료: {filepath}")
    
    def validate_dataset(self, dataset: List[Tuple[str, str, str]]) -> Dict:
        """데이터셋 검증"""
        total_count = len(dataset)
        
        # 상황 유형 분포
        situation_types = {"극도위험": 0, "위험상황": 0, "초과상황": 0, "정상상황": 0, "기준없음": 0}
        
        # 감지 유형 분포
        detection_types = {"연기": 0, "화염": 0, "불꽃": 0, "복합": 0}
        
        # 감지 시간 분포
        time_ranges = {"단기(2-5초)": 0, "중기(6-10초)": 0, "장기(11-20초)": 0, "초장기(21초+)": 0}
        
        # 문장 길이 분포
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        
        for input_str, output_str, domain in dataset:
            # 상황 유형 분류
            if "극도" in output_str:
                situation_types["극도위험"] += 1
            elif "위험" in output_str and ("수준" in output_str or "상황" in output_str):
                situation_types["위험상황"] += 1
            elif "초과" in output_str:
                situation_types["초과상황"] += 1
            elif "정상" in output_str or "안전한" in output_str:
                situation_types["정상상황"] += 1
            else:
                situation_types["기준없음"] += 1
            
            # 감지 유형 분류
            if "연기·화염" in input_str or "불꽃·연기" in input_str or "화염·연기" in input_str:
                detection_types["복합"] += 1
            elif "연기" in input_str:
                detection_types["연기"] += 1
            elif "화염" in input_str:
                detection_types["화염"] += 1
            elif "불꽃" in input_str:
                detection_types["불꽃"] += 1
            
            # 감지 시간 추출
            time_match = re.search(r'(\d+)초', input_str)
            if time_match:
                detection_time = int(time_match.group(1))
                if detection_time <= 5:
                    time_ranges["단기(2-5초)"] += 1
                elif detection_time <= 10:
                    time_ranges["중기(6-10초)"] += 1
                elif detection_time <= 20:
                    time_ranges["장기(11-20초)"] += 1
                else:
                    time_ranges["초장기(21초+)"] += 1
            
            # 문장 길이 체크 (한국어 종결어미 핵심 패턴 기반)
            sentence_endings_patterns = [
                r'다\.',  # ~다. (습니다, 됩니다 등)
                r'요\.',  # ~요. (해주세요 등)  
                r'오\.',  # ~오. (하십시오 등)
            ]
            
            matched_positions = set()
            sentence_count = 0
            
            for pattern in sentence_endings_patterns:
                matches = re.finditer(pattern, output_str)
                for match in matches:
                    pos = match.end()
                    if pos not in matched_positions:
                        matched_positions.add(pos)
                        sentence_count += 1
            
            if sentence_count == 2:
                sentence_lengths["2문장"] += 1
            elif sentence_count == 3:
                sentence_lengths["3문장"] += 1
            else:
                sentence_lengths["기타"] += 1
        
        return {
            "total_count": total_count,
            "situation_types": {
                "극도위험": f"{situation_types['극도위험']} ({situation_types['극도위험']/total_count*100:.1f}%)",
                "위험상황": f"{situation_types['위험상황']} ({situation_types['위험상황']/total_count*100:.1f}%)",
                "초과상황": f"{situation_types['초과상황']} ({situation_types['초과상황']/total_count*100:.1f}%)",
                "정상상황": f"{situation_types['정상상황']} ({situation_types['정상상황']/total_count*100:.1f}%)",
                "기준없음": f"{situation_types['기준없음']} ({situation_types['기준없음']/total_count*100:.1f}%)"
            },
            "detection_types": {
                "연기": f"{detection_types['연기']} ({detection_types['연기']/total_count*100:.1f}%)",
                "화염": f"{detection_types['화염']} ({detection_types['화염']/total_count*100:.1f}%)",
                "불꽃": f"{detection_types['불꽃']} ({detection_types['불꽃']/total_count*100:.1f}%)",
                "복합": f"{detection_types['복합']} ({detection_types['복합']/total_count*100:.1f}%)"
            },
            "time_ranges": {
                "단기(2-5초)": f"{time_ranges['단기(2-5초)']} ({time_ranges['단기(2-5초)']/total_count*100:.1f}%)",
                "중기(6-10초)": f"{time_ranges['중기(6-10초)']} ({time_ranges['중기(6-10초)']/total_count*100:.1f}%)",
                "장기(11-20초)": f"{time_ranges['장기(11-20초)']} ({time_ranges['장기(11-20초)']/total_count*100:.1f}%)",
                "초장기(21초+)": f"{time_ranges['초장기(21초+)']} ({time_ranges['초장기(21초+)']/total_count*100:.1f}%)"
            },
            "sentence_lengths": {
                "2문장": f"{sentence_lengths['2문장']} ({sentence_lengths['2문장']/total_count*100:.1f}%)",
                "3문장": f"{sentence_lengths['3문장']} ({sentence_lengths['3문장']/total_count*100:.1f}%)",
                "기타": f"{sentence_lengths['기타']} ({sentence_lengths['기타']/total_count*100:.1f}%)"
            }
        }

if __name__ == "__main__":
    """
    메인 실행 부분
    - 1000개 데이터셋 생성
    - CSV 파일 저장
    - 검증 결과 출력
    """
    # 데이터셋 생성기 인스턴스 생성
    generator = Domain3SmokeFlameDetectionGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 저장 경로 설정
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain3_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 데이터셋 검증 및 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인3 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n⚠️ 상황 유형 분포:")
    print(f"   극도위험: {validation_results['situation_types']['극도위험']}")
    print(f"   위험상황: {validation_results['situation_types']['위험상황']}")
    print(f"   초과상황: {validation_results['situation_types']['초과상황']}")
    print(f"   정상상황: {validation_results['situation_types']['정상상황']}")
    print(f"   기준없음: {validation_results['situation_types']['기준없음']}")
    print(f"\n🔥 감지 유형 분포:")
    print(f"   연기: {validation_results['detection_types']['연기']}")
    print(f"   화염: {validation_results['detection_types']['화염']}")
    print(f"   불꽃: {validation_results['detection_types']['불꽃']}")
    print(f"   복합: {validation_results['detection_types']['복합']}")
    print(f"\n⏱️ 감지 시간 분포:")
    print(f"   단기(2-5초): {validation_results['time_ranges']['단기(2-5초)']}")
    print(f"   중기(6-10초): {validation_results['time_ranges']['중기(6-10초)']}")
    print(f"   장기(11-20초): {validation_results['time_ranges']['장기(11-20초)']}")
    print(f"   초장기(21초+): {validation_results['time_ranges']['초장기(21초+)']}")
    print(f"\n📝 문장 길이 분포:")
    print(f"   2문장: {validation_results['sentence_lengths']['2문장']}")
    print(f"   3문장: {validation_results['sentence_lengths']['3문장']}")
    print(f"   기타: {validation_results['sentence_lengths']['기타']}")