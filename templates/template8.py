import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain8HybridGenerator:
    """
    도메인8 히트맵 기반 체류 위험구간 분석 데이터셋 생성기
    
    주요 기능:
    - 대규모 시설의 사람 밀집 위험 관리 데이터 생성 (최대 2000명)
    - 7가지 패턴(A~G형)으로 다양한 상황 시뮬레이션
    - 장소별 적절한 구역명 매핑
    - 중복 단어 최소화 및 문어체 다양화
    - 2~3문장 자연스러운 흐름의 Output 생성
    - Input-Output 정보 완벽 매칭
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 인원 규모별 설정 (3~2000명 범위)
        self.person_ranges = {
            "소규모": {"min": 3, "max": 20, "weight": 25},     # 25%
            "중규모": {"min": 21, "max": 100, "weight": 40},   # 40%
            "대규모": {"min": 101, "max": 500, "weight": 25},  # 25%
            "초대규모": {"min": 501, "max": 2000, "weight": 10} # 10%
        }
        
        # 체류시간 범위 (분)
        self.stay_time_ranges = {
            "단기": {"min": 5, "max": 20, "weight": 40},     # 5-20분, 40% 확률
            "중기": {"min": 21, "max": 35, "weight": 45},    # 21-35분, 45% 확률
            "장기": {"min": 36, "max": 50, "weight": 15}     # 36-50분, 15% 확률
        }
        
        # 밀도 지수 범위 (0.35~0.98)
        self.density_ranges = {
            "낮음": {"min": 0.35, "max": 0.60, "weight": 20},   # 낮은 밀도
            "보통": {"min": 0.61, "max": 0.80, "weight": 50},   # 보통 밀도
            "높음": {"min": 0.81, "max": 0.98, "weight": 30}    # 높은 밀도
        }
        
        # 장소명 (70%) - 대규모 시설, 사람 밀집 위험 관리 필요 장소
        self.locations = [
            # 교통시설
            "지하철역", "버스터미널", "공항", "기차역", "고속터미널", "항만터미널",
            
            # 상업시설  
            "백화점", "대형마트", "쇼핑몰", "아울렛", "복합쇼핑센터", "전통시장",
            
            # 문화/체육시설
            "콘서트홀", "야구장", "축구경기장", "농구경기장", "체육관", "경기장", 
            "박물관", "미술관", "전시관", "문화회관", "공연장", "극장",
            
            # 교육시설
            "대학교", "도서관", "컨벤션센터", "전시센터",
            
            # 의료시설
            "종합병원", "대학병원", "의료센터",
            
            # 공공시설
            "시청", "구청", "관공서", "법원", "우체국", "은행본점",
            
            # 레저시설
            "테마파크", "놀이공원", "워터파크", "스키장", "리조트",
            
            # 기타 대규모 시설
            "호텔", "컨퍼런스센터", "전시컨벤션센터", "국제회의장"
        ]
        
        # 공통 구역명 (모든 장소에 적용 가능)
        self.common_areas = [
            "A구역", "B구역", "C구역", "D구역", "E구역", "F구역", "G구역", "H구역",
            "1층", "2층", "3층", "4층", "5층", "지하1층", "지하2층", "지하3층",
            "입구", "로비", "대기실", "휴게실", "북관", "남관", "동관", "서관"
        ]
        
        # 장소별 전용 구역명 매핑
        self.location_specific_areas = {
            # 교통시설
            "지하철역": ["대합실", "플랫폼", "환승통로", "출입구", "매표소"],
            "버스터미널": ["대합실", "승차장", "하차장", "매표소", "대기실"],
            "공항": ["출발게이트", "도착게이트", "보안검색대", "수하물찾는곳", "체크인카운터", "면세구역", "출국장", "입국장"],
            "기차역": ["승강장", "대합실", "매표소", "출입구"],
            "고속터미널": ["승차장", "하차장", "대합실", "매표소"],
            "항만터미널": ["승선장", "하선장", "대기실", "매표소"],
            
            # 상업시설
            "백화점": ["푸드코트", "계산대구역", "중앙광장", "아트리움", "명품관", "생활관", "식품관", "의류관", "화장품관"],
            "대형마트": ["계산대구역", "식품관", "생활용품관", "의류관", "전자제품관"],
            "쇼핑몰": ["푸드코트", "중앙광장", "아트리움", "계산대구역"],
            "아울렛": ["푸드코트", "중앙광장", "계산대구역"],
            "복합쇼핑센터": ["푸드코트", "중앙광장", "아트리움", "계산대구역"],
            "전통시장": ["중앙통로", "입구", "먹거리코너"],
            
            # 문화/체육시설
            "콘서트홀": ["관람석", "입장게이트", "매표소", "VIP라운지", "무대앞"],
            "야구장": ["관람석", "입장게이트", "매표소", "내야석", "외야석", "VIP석"],
            "축구경기장": ["관람석", "입장게이트", "매표소", "응원석"],
            "농구경기장": ["관람석", "입장게이트", "매표소"],
            "체육관": ["관람석", "입장게이트", "매표소"],
            "경기장": ["관람석", "입장게이트", "매표소"],
            "박물관": ["상설전시실", "기획전시실", "매표소", "뮤지엄샵"],
            "미술관": ["상설전시실", "기획전시실", "매표소", "뮤지엄샵"],
            "전시관": ["전시실", "매표소", "컨벤션홀"],
            "문화회관": ["대극장", "소극장", "매표소", "로비"],
            "공연장": ["관람석", "매표소", "VIP라운지"],
            "극장": ["상영관", "매표소", "매점"],
            
            # 교육시설
            "대학교": ["학생식당", "도서관", "강의동", "학생회관", "기숙사"],
            "도서관": ["열람실", "자료실", "대출반납데스크"],
            "컨벤션센터": ["컨벤션홀", "전시실", "회의실"],
            "전시센터": ["전시홀", "컨벤션룸"],
            
            # 의료시설
            "종합병원": ["외래진료센터", "응급실", "입원병동", "검사실", "약국"],
            "대학병원": ["외래진료센터", "응급실", "입원병동", "검사실"],
            "의료센터": ["진료실", "대기실", "검사실"],
            
            # 공공시설
            "시청": ["민원실", "대회의실", "로비"],
            "구청": ["민원실", "대회의실", "로비"],
            "관공서": ["민원실", "대기실", "로비"],
            "법원": ["법정", "대기실", "민원실"],
            "우체국": ["창구", "대기실"],
            "은행본점": ["창구", "대기실", "VIP라운지"],
            
            # 레저시설
            "테마파크": ["입장게이트", "푸드코트", "기념품샵", "놀이기구대기줄"],
            "놀이공원": ["입장게이트", "푸드코트", "기념품샵", "놀이기구대기줄"],
            "워터파크": ["입장게이트", "탈의실", "휴게실", "풀사이드"],
            "스키장": ["리프트승강장", "렌탈샵", "휴게실"],
            "리조트": ["로비", "레스토랑", "컨벤션홀", "수영장"],
            
            # 기타 대규모 시설
            "호텔": ["로비", "레스토랑", "컨벤션홀", "연회장"],
            "컨퍼런스센터": ["컨벤션홀", "회의실", "로비"],
            "전시컨벤션센터": ["전시홀", "컨벤션룸", "로비"],
            "국제회의장": ["대회의실", "소회의실", "로비"]
        }
        
        # 7가지 패턴 정의 (A~G형)
        self.patterns = {
            "A": {"name": "현재상황분석형", "weight": 30},      # 현재 상황 분석 중심
            "B": {"name": "단기예측분석형", "weight": 20},      # 2-8시간 후 예측
            "C": {"name": "내일예측분석형", "weight": 15},      # 내일 동시간대 예측
            "D": {"name": "위험감지즉시형", "weight": 15},      # 즉시 위험 감지 및 조치
            "E": {"name": "정상확인관리형", "weight": 10},      # 정상 상황 확인 및 관리
            "F": {"name": "패턴분석예측형", "weight": 7},       # 패턴 분석 기반 예측
            "G": {"name": "미설정예측형", "weight": 3}         # 기타 예측 상황
        }
        
        # 조치 표현 유형별 설정
        self.action_types = {
            "즉시대응": {
                "phrases": [
                    "긴급 분산 조치를 실시하십시오", "즉시 동선 분산 조치가 필요합니다",
                    "해당 구역 과밀 해소 조치를 취하십시오", "즉시 과밀 상황 해결이 요구됩니다",
                    "긴급 대응 조치를 시행해주세요", "즉각적인 밀도 완화 조치를 하십시오"
                ],
                "weight": 40
            },
            "사전대응": {
                "phrases": [
                    "사전 동선 분산 조치를 준비하십시오", "예방적 동선 안내를 실시해주세요",
                    "미리 과밀 방지 조치를 취하십시오", "사전 대응 체계를 가동해주세요",
                    "예방 차원의 분산 유도가 필요합니다", "선제적 밀도 관리를 시행하십시오"
                ],
                "weight": 30
            },
            "지속관리": {
                "phrases": [
                    "지속적인 동향 관리가 필요합니다", "체류 동향 관리를 지속하십시오",
                    "계속적인 상황 모니터링을 해주세요", "지속 관찰 체계를 유지하십시오",
                    "꾸준한 밀도 추이 관리가 요구됩니다", "연속적인 상황 점검을 실시해주세요"
                ],
                "weight": 20
            },
            "단계별대응": {
                "phrases": [
                    "단계별 분산 조치를 시행하십시오", "순차적 동선 유도를 실시해주세요",
                    "체계적인 밀도 완화를 진행하십시오", "단계적 과밀 해소를 시행해주세요",
                    "순서대로 분산 유도를 실시하십시오", "체계적 대응 절차를 가동해주세요"
                ],
                "weight": 10
            }
        }
        
        # 문어체 어미 다양화 (중복 최소화)
        self.sentence_endings = [
            "습니다", "해주세요", "하십시오", "됩니다", "가 필요합니다", 
            "을 권장합니다", "바랍니다", "을 요청합니다", "을 권고합니다", 
            "을 제안합니다", "이 요구됩니다", "을 실시하십시오"
        ]
        
        # 상황 분석 표현 다양화 (중복 단어 최소화)
        self.situation_expressions = {
            "over_capacity": [
                "기준 인원을 {diff}명 초과했습니다", "허용 인원보다 {diff}명 많습니다",
                "정원을 {diff}명 넘어섰습니다", "기준치보다 {diff}명 높습니다",
                "설정 인원 대비 {diff}명 증가했습니다", "표준 인원에서 {diff}명 상회했습니다"
            ],
            "normal_range": [
                "기준 인원 내에서 운영되고 있습니다", "정상 범위에서 관리되고 있습니다",
                "허용 인원 내에서 유지되고 있습니다", "적정 수준에서 운영 중입니다",
                "기준치 내에서 관리되고 있습니다", "정원 범위에서 운영되고 있습니다"
            ],
            "prediction": [
                "분석됩니다", "예측됩니다", "판단됩니다", "추정됩니다", 
                "전망됩니다", "예상됩니다", "산출됩니다"
            ]
        }
        
        # 이유/근거 표현 (문장 연결용) - 대규모 상황 반영
        self.reason_expressions = [
            "기준 초과로 인해 사고 위험이 높습니다", 
            "과밀 상황으로 안전사고 우려가 있습니다",
            "밀도 증가로 위험 상황이 예상됩니다", 
            "인원 집중으로 사고 발생 가능성이 있습니다",
            "체류 인원 증가로 안전 관리가 필요합니다", 
            "과밀도로 인한 위험 요소가 감지됩니다",
            "대규모 인원 집중으로 긴급 대응이 필요합니다",
            "초과 인원으로 인한 안전사고 위험이 증가했습니다",
            "다수 인원 체류로 비상상황 대비가 요구됩니다"
        ]
        
        # 연결어 패턴 (문장 순서 다양화)
        self.connectors = [
            "따라서", "이에", "그러므로", "이로 인해", "때문에"
        ]
        
    def generate_person_count(self) -> int:
        """
        인원 수 생성 함수
        
        Returns:
            int: 3~2000명 범위의 동시 체류 인원
        """
        # 가중치에 따른 규모 선택
        weights = [self.person_ranges[key]["weight"] for key in self.person_ranges.keys()]
        selected_range = random.choices(list(self.person_ranges.keys()), weights=weights)[0]
        
        # 선택된 범위에서 인원 수 생성
        range_info = self.person_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_stay_time(self) -> int:
        """
        체류시간 생성 함수
        
        Returns:
            int: 5~50분 범위의 체류시간
        """
        # 가중치에 따른 시간대 선택
        weights = [self.stay_time_ranges[key]["weight"] for key in self.stay_time_ranges.keys()]
        selected_range = random.choices(list(self.stay_time_ranges.keys()), weights=weights)[0]
        
        # 선택된 범위에서 체류시간 생성
        range_info = self.stay_time_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_density(self) -> float:
        """
        밀도 지수 생성 함수
        
        Returns:
            float: 0.35~0.98 범위의 밀도 지수 (소수점 둘째자리)
        """
        # 가중치에 따른 밀도 범위 선택
        weights = [self.density_ranges[key]["weight"] for key in self.density_ranges.keys()]
        selected_range = random.choices(list(self.density_ranges.keys()), weights=weights)[0]
        
        # 선택된 범위에서 밀도 생성
        range_info = self.density_ranges[selected_range]
        density = random.uniform(range_info["min"], range_info["max"])
        return round(density, 2)
    
    def generate_location(self) -> str:
        """
        장소 생성 함수 - 장소에 맞는 구역명 매핑
        
        Returns:
            str: 장소명(70%) 또는 장소명+적절한구역명(30%)
        """
        location = random.choice(self.locations)
        
        if random.random() < 0.7:
            # 장소명만 (70%)
            return location
        else:
            # 장소명 + 해당 장소에 맞는 구역명 (30%)
            area = self.get_appropriate_area(location)
            return f"{location} {area}"
    
    def get_appropriate_area(self, location: str) -> str:
        """
        장소에 맞는 적절한 구역명 반환
        
        Args:
            location (str): 장소명
            
        Returns:
            str: 해당 장소에 적절한 구역명
        """
        # 해당 장소의 전용 구역명이 있는 경우 70% 확률로 사용
        if location in self.location_specific_areas and random.random() < 0.7:
            return random.choice(self.location_specific_areas[location])
        else:
            # 공통 구역명 사용 (30% 확률 또는 전용 구역명이 없는 경우)
            return random.choice(self.common_areas)
    
    def generate_time_format(self) -> str:
        """
        시간 형식 생성 함수
        
        Returns:
            str: HH:MM 형식(50%) 또는 한글 시간(50%)
        """
        hour = random.randint(6, 23)  # 06:00 ~ 23:59
        minute = random.randint(0, 59)
        
        if random.random() < 0.5:
            # HH:MM 형식 (50%)
            return f"{hour:02d}:{minute:02d}"
        else:
            # 한글 시간 형식 (50%)
            if hour < 12:
                if hour == 0:
                    return f"자정 12시 {minute}분"
                else:
                    return f"오전 {hour}시 {minute}분"
            elif hour == 12:
                return f"정오 12시 {minute}분"
            else:
                return f"오후 {hour-12}시 {minute}분"
    
    def generate_coordinates(self) -> str:
        """
        좌표 정보 생성 함수 (30% 확률)
        
        Returns:
            str: 좌표 정보 또는 빈 문자열
        """
        if random.random() < 0.3:
            x = random.randint(100, 600)
            y = random.randint(100, 500)
            return f"(X:{x} Y:{y}) "
        return ""
    
    def select_pattern(self) -> str:
        """
        패턴 선택 함수
        
        Returns:
            str: 선택된 패턴 (A~G)
        """
        patterns = list(self.patterns.keys())
        weights = [self.patterns[pattern]["weight"] for pattern in patterns]
        return random.choices(patterns, weights=weights)[0]
    
    def generate_baseline_person_count(self, current_count: int) -> int:
        """
        기준 인원 생성 함수 (2000명까지 대응)
        
        Args:
            current_count (int): 현재 인원 수
            
        Returns:
            int: 기준 인원 수
        """
        # 현재 인원보다 적게 설정 (위험 상황 시뮬레이션)
        if current_count <= 20:
            return max(3, current_count - random.randint(1, 5))
        elif current_count <= 100:
            return current_count - random.randint(5, 25)
        elif current_count <= 500:
            return current_count - random.randint(20, 80)
        else:  # 501명 이상 (초대규모)
            return current_count - random.randint(50, 300)
    
    def generate_input(self) -> Dict:
        """
        Input 데이터 생성 함수
        
        Returns:
            Dict: Input 데이터 딕셔너리
        """
        location = self.generate_location()
        time = self.generate_time_format()
        coordinates = self.generate_coordinates()
        person_count = self.generate_person_count()
        stay_time = self.generate_stay_time()
        density = self.generate_density()
        baseline_count = self.generate_baseline_person_count(person_count)
        
        return {
            "location": location,
            "time": time,
            "coordinates": coordinates,
            "person_count": person_count,
            "stay_time": stay_time,
            "density": density,
            "baseline_count": baseline_count
        }
    
    def generate_situation_analysis(self, input_data: Dict, pattern: str) -> str:
        """
        상황 분석 문장 생성 함수 (자연스러운 시작)
        
        Args:
            input_data (Dict): Input 데이터
            pattern (str): 패턴 타입
            
        Returns:
            str: 상황 분석 문장
        """
        location = input_data["location"]
        time = input_data["time"]
        person_count = input_data["person_count"]
        stay_time = input_data["stay_time"]
        density = input_data["density"]
        baseline_count = input_data["baseline_count"]
        
        # 자연스러운 시작점 선택 (장소, 장소+구역명, 시간)
        start_options = [
            f"{location}에서",  # 장소로 시작
            f"{time} {location}에서",  # 시간+장소로 시작
            f"{time} 현재 {location}에서"  # 시간+현재+장소로 시작
        ]
        
        start_phrase = random.choice(start_options)
        
        # 기본 상황 정보
        base_info = f"{start_phrase} 동시 {person_count}명이 체류시간 {stay_time}분 밀도 {density}로"
        
        # 패턴별 분석 추가
        if pattern in ["A", "D"]:  # 현재상황분석형, 위험감지즉시형
            diff = person_count - baseline_count
            if diff > 0:
                over_expr = random.choice(self.situation_expressions["over_capacity"]).format(diff=diff)
                return f"{base_info} 기준 인원 {baseline_count}명을 {over_expr}"
            else:
                normal_expr = random.choice(self.situation_expressions["normal_range"])
                return f"{base_info} {normal_expr}"
                
        elif pattern in ["B", "F"]:  # 단기예측분석형, 패턴분석예측형
            pred_expr = random.choice(self.situation_expressions["prediction"])
            # 인원 규모에 따른 증가폭 조정
            if person_count <= 50:
                pred_count = person_count + random.randint(3, 15)
            elif person_count <= 200:
                pred_count = person_count + random.randint(10, 50)
            else:
                pred_count = person_count + random.randint(20, 100)
            pred_time = random.randint(2, 8)
            return f"{base_info} 기준 인원 {baseline_count}명 대비 {pred_time}시간 후 {pred_count}명까지 증가할 것으로 {pred_expr}"
            
        elif pattern == "C":  # 내일예측분석형
            pred_expr = random.choice(self.situation_expressions["prediction"])
            # 인원 규모에 따른 변동폭 조정
            if baseline_count <= 50:
                pred_count = baseline_count + random.randint(-10, 20)
            elif baseline_count <= 200:
                pred_count = baseline_count + random.randint(-30, 60)
            else:
                pred_count = baseline_count + random.randint(-50, 150)
            return f"{base_info} 기준 인원 {baseline_count}명 대비 내일 동시간대 {pred_count}명 수준으로 {pred_expr}"
            
        elif pattern == "E":  # 정상확인관리형
            normal_expr = random.choice(self.situation_expressions["normal_range"])
            return f"{base_info} 기준 인원 {baseline_count}명 범위에서 {normal_expr}"
            
        else:  # G형 (미설정예측형)
            pred_expr = random.choice(self.situation_expressions["prediction"])
            return f"{base_info} 기준 인원 {baseline_count}명 대비 향후 변동이 {pred_expr}"
    
    def generate_action_phrase(self, pattern: str) -> str:
        """
        조치 문구 생성 함수
        
        Args:
            pattern (str): 패턴 타입
            
        Returns:
            str: 조치 문구
        """
        if pattern in ["A", "D"]:  # 즉시 대응
            action_type = "즉시대응"
        elif pattern in ["B", "C", "F"]:  # 사전 대응
            action_type = "사전대응"
        elif pattern == "E":  # 지속 관리
            action_type = "지속관리"
        else:  # G형 - 단계별 대응
            action_type = "단계별대응"
        
        # 가중치에 따른 조치 유형 선택
        if random.random() * 100 <= self.action_types[action_type]["weight"]:
            return random.choice(self.action_types[action_type]["phrases"])
        else:
            # 다른 조치 유형에서 랜덤 선택
            other_types = [t for t in self.action_types.keys() if t != action_type]
            selected_type = random.choice(other_types)
            return random.choice(self.action_types[selected_type]["phrases"])
    
    def generate_natural_flow_output(self, input_data: Dict, pattern: str) -> str:
        """
        자연스러운 흐름의 2~3문장 Output 생성 함수
        
        Args:
            input_data (Dict): Input 데이터
            pattern (str): 패턴 타입
            
        Returns:
            str: 완성된 Output 문장 (2~3문장)
        """
        # 기본 구성 요소 생성
        situation_analysis = self.generate_situation_analysis(input_data, pattern)
        action_phrase = self.generate_action_phrase(pattern)
        
        # 문장 길이 결정 (2문장: 60%, 3문장: 40%)
        sentence_count = random.choices([2, 3], weights=[60, 40])[0]
        
        if sentence_count == 2:
            # 2문장 패턴: [상황] + [조치]
            return f"{situation_analysis}. {action_phrase}."
        
        else:
            # 3문장 패턴 선택
            patterns_3sentence = [
                # 패턴 1: [상황] + [이유] + [조치] (50%)
                lambda: f"{situation_analysis}. {random.choice(self.reason_expressions)}. {random.choice(self.connectors)} {action_phrase}.",
                
                # 패턴 2: [상황] + [조치] + [근거] (30%)  
                lambda: f"{situation_analysis}. {action_phrase}. {random.choice(self.reason_expressions)}.",
                
                # 패턴 3: [상황] + [연결어] + [조치] (20%)
                lambda: f"{situation_analysis}. {random.choice(self.connectors)} {action_phrase}. 현재 상황을 면밀히 모니터링하십시오."
            ]
            
            # 가중치에 따른 3문장 패턴 선택
            weights_3sentence = [50, 30, 20]
            selected_pattern = random.choices(patterns_3sentence, weights=weights_3sentence)[0]
            
            return selected_pattern()
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """
        단일 데이터 생성 함수
        
        Returns:
            Tuple[str, str, str]: (Input, Output, Domain)
        """
        # Input 데이터 생성
        input_data = self.generate_input()
        
        # 패턴 선택
        pattern = self.select_pattern()
        
        # Input 문자열 구성
        coordinates = input_data["coordinates"]
        input_str = (f"{coordinates}{input_data['time']} {input_data['location']}에서 "
                    f"동시 {input_data['person_count']}명 체류시간 {input_data['stay_time']}분 "
                    f"밀도 {input_data['density']}, 기준 인원 {input_data['baseline_count']}명")
        
        # Output 생성 (자연스러운 흐름의 2~3문장)
        output_str = self.generate_natural_flow_output(input_data, pattern)
        
        # 도메인명
        domain = "히트맵 기반 체류 위험구간 분석"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """
        데이터셋 생성 함수
        
        Args:
            count (int): 생성할 데이터 개수
            
        Returns:
            List[Tuple[str, str, str]]: 생성된 데이터셋 리스트
        """
        dataset = []
        
        print(f"🔄 도메인8 데이터셋 {count}개 생성 중...")
        
        for i in range(count):
            if (i + 1) % 100 == 0:
                print(f"   진행률: {i + 1}/{count} ({((i + 1)/count)*100:.1f}%)")
            
            data = self.generate_single_data()
            dataset.append(data)
        
        print(f"✅ 데이터셋 생성 완료: {count}개")
        return dataset
    
    def save_to_csv(self, dataset: List[Tuple[str, str, str]], filepath: str):
        """
        CSV 파일 저장 함수 (쌍따옴표 문제 해결)
        
        Args:
            dataset (List[Tuple[str, str, str]]): 데이터셋
            filepath (str): 저장 경로
        """
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # CSV 파일 저장 (모든 필드를 쌍따옴표로 감싸기)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            
            # 헤더 작성
            writer.writerow(['Input', 'Output', 'Domain'])
            
            # 데이터 작성
            for input_str, output_str, domain in dataset:
                writer.writerow([input_str, output_str, domain])
        
        print(f"💾 CSV 파일 저장 완료: {filepath}")
    
    def validate_dataset(self, dataset: List[Tuple[str, str, str]]) -> Dict:
        """
        데이터셋 검증 함수
        
        Args:
            dataset (List[Tuple[str, str, str]]): 데이터셋
            
        Returns:
            Dict: 검증 결과
        """
        # 패턴별 분포 계산
        pattern_distribution = Counter()
        
        # 위험도 분류 계산
        risk_categories = {"위험/어색/침고": 0, "예측/예시": 0, "정상/관리": 0}
        
        # 구조형/설명형 비율 계산
        input_types = {"구조형": 0, "설명형": 0}
        
        # 장소 vs 장소+구역명 비율 계산
        location_types = {"장소만": 0, "장소+구역명": 0}
        
        # 좌표 정보 사용률 계산
        coordinate_usage = {"사용": 0, "미사용": 0}
        
        # 시간 형식 비율 계산
        time_formats = {"HH:MM": 0, "한글시간": 0}
        
        # 문장 길이 분포 계산
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        
        # 인원 규모 분포 계산
        person_ranges = {"소규모": 0, "중규모": 0, "대규모": 0, "초대규모": 0}
        
        for input_str, output_str, domain in dataset:
            # 좌표 정보 체크
            if "(X:" in input_str:
                coordinate_usage["사용"] += 1
            else:
                coordinate_usage["미사용"] += 1
            
            # 시간 형식 체크
            if re.search(r'\d{2}:\d{2}', input_str):
                time_formats["HH:MM"] += 1
            else:
                time_formats["한글시간"] += 1
            
            # 장소 형식 체크 (간단한 휴리스틱)
            location_part = input_str.split("에서")[0]
            if any(area in location_part for area in self.common_areas) or \
               any(area in location_part for area_list in self.location_specific_areas.values() for area in area_list):
                location_types["장소+구역명"] += 1
            else:
                location_types["장소만"] += 1
            
            # 위험도 분류 (Output 내용 기반)
            if any(word in output_str for word in ["초과", "넘어", "높습니다", "즉시", "긴급"]):
                risk_categories["위험/어색/침고"] += 1
            elif any(word in output_str for word in ["예측", "분석", "판단", "전망", "예상"]):
                risk_categories["예측/예시"] += 1
            else:
                risk_categories["정상/관리"] += 1
            
            # 문장 길이 체크 (한국어 종결어미 핵심 패턴 기반)
            # 한국어 문장 종결의 핵심 패턴: 다., 요., 오.
            sentence_endings_patterns = [
                r'다\.',  # ~다. (습니다, 됩니다, 판단됩니다 등)
                r'요\.',  # ~요. (해주세요, 유지해주세요 등)  
                r'오\.',  # ~오. (하십시오, 실시하십시오 등)
            ]
            
            # 중복 방지를 위해 이미 매칭된 위치 추적
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
            
            # 인원 규모 체크
            person_match = re.search(r'동시 (\d+)명', input_str)
            if person_match:
                person_count = int(person_match.group(1))
                if person_count <= 20:
                    person_ranges["소규모"] += 1
                elif person_count <= 100:
                    person_ranges["중규모"] += 1
                elif person_count <= 500:
                    person_ranges["대규모"] += 1
                else:
                    person_ranges["초대규모"] += 1
        
        total_count = len(dataset)
        
        return {
            "total_count": total_count,
            "coordinate_usage": {
                "사용": f"{coordinate_usage['사용']} ({coordinate_usage['사용']/total_count*100:.1f}%)",
                "미사용": f"{coordinate_usage['미사용']} ({coordinate_usage['미사용']/total_count*100:.1f}%)"
            },
            "time_formats": {
                "HH:MM": f"{time_formats['HH:MM']} ({time_formats['HH:MM']/total_count*100:.1f}%)",
                "한글시간": f"{time_formats['한글시간']} ({time_formats['한글시간']/total_count*100:.1f}%)"
            },
            "location_types": {
                "장소만": f"{location_types['장소만']} ({location_types['장소만']/total_count*100:.1f}%)",
                "장소+구역명": f"{location_types['장소+구역명']} ({location_types['장소+구역명']/total_count*100:.1f}%)"
            },
            "risk_categories": {
                "위험/어색/침고": f"{risk_categories['위험/어색/침고']} ({risk_categories['위험/어색/침고']/total_count*100:.1f}%)",
                "예측/예시": f"{risk_categories['예측/예시']} ({risk_categories['예측/예시']/total_count*100:.1f}%)",
                "정상/관리": f"{risk_categories['정상/관리']} ({risk_categories['정상/관리']/total_count*100:.1f}%)"
            },
            "sentence_lengths": {
                "2문장": f"{sentence_lengths['2문장']} ({sentence_lengths['2문장']/total_count*100:.1f}%)",
                "3문장": f"{sentence_lengths['3문장']} ({sentence_lengths['3문장']/total_count*100:.1f}%)",
                "기타": f"{sentence_lengths['기타']} ({sentence_lengths['기타']/total_count*100:.1f}%)"
            },
            "person_ranges": {
                "소규모(3-20명)": f"{person_ranges['소규모']} ({person_ranges['소규모']/total_count*100:.1f}%)",
                "중규모(21-100명)": f"{person_ranges['중규모']} ({person_ranges['중규모']/total_count*100:.1f}%)",
                "대규모(101-500명)": f"{person_ranges['대규모']} ({person_ranges['대규모']/total_count*100:.1f}%)",
                "초대규모(501-2000명)": f"{person_ranges['초대규모']} ({person_ranges['초대규모']/total_count*100:.1f}%)"
            }
        }

if __name__ == "__main__":
    """
    메인 실행 부분
    - 1000개 데이터셋 생성
    - 새로운 CSV 파일 저장
    - 검증 결과 출력
    """
    # 데이터셋 생성기 인스턴스 생성
    generator = Domain8HybridGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 새로운 파일명으로 저장 경로 설정
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain8_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 데이터셋 검증 및 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인8 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n📍 좌표 정보 사용률:")
    print(f"   사용: {validation_results['coordinate_usage']['사용']}")
    print(f"   미사용: {validation_results['coordinate_usage']['미사용']}")
    print(f"\n⏰ 시간 형식 분포:")
    print(f"   HH:MM 형식: {validation_results['time_formats']['HH:MM']}")
    print(f"   한글 시간: {validation_results['time_formats']['한글시간']}")
    print(f"\n🏢 장소 형식 분포:")
    print(f"   장소만: {validation_results['location_types']['장소만']}")
    print(f"   장소+구역명: {validation_results['location_types']['장소+구역명']}")
    print(f"\n⚠️ 위험도 분류 분포:")
    print(f"   위험/어색/침고: {validation_results['risk_categories']['위험/어색/침고']}")
    print(f"   예측/예시: {validation_results['risk_categories']['예측/예시']}")
    print(f"   정상/관리: {validation_results['risk_categories']['정상/관리']}")
    print(f"\n📝 문장 길이 분포:")
    print(f"   2문장: {validation_results['sentence_lengths']['2문장']}")
    print(f"   3문장: {validation_results['sentence_lengths']['3문장']}")
    print(f"   기타: {validation_results['sentence_lengths']['기타']}")
    print(f"\n👥 인원 규모 분포:")
    print(f"   소규모(3-20명): {validation_results['person_ranges']['소규모(3-20명)']}")
    print(f"   중규모(21-100명): {validation_results['person_ranges']['중규모(21-100명)']}")
    print(f"   대규모(101-500명): {validation_results['person_ranges']['대규모(101-500명)']}")
    print(f"   초대규모(501-2000명): {validation_results['person_ranges']['초대규모(501-2000명)']}")