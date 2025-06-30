import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain2FallDetectionGenerator:
    """
    도메인2 쓰러짐 및 장기 정지 감지 데이터셋 생성기
    
    주요 기능:
    - 다양한 장소의 쓰러짐 및 장기 정지 상황 데이터 생성
    - 인원별 자세/상태와 지속시간 분석
    - 기준시간 대비 위험도 판정
    - 상황별 적절한 조치 방안 제시
    - 자연스러운 2~3문장 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 인원 규모별 설정 (1~15명 범위)
        self.person_ranges = {
            "개인": {"min": 1, "max": 1, "weight": 40},        # 40%
            "소수": {"min": 2, "max": 3, "weight": 30},        # 30%
            "중간": {"min": 4, "max": 6, "weight": 25},        # 25%
            "다수": {"min": 7, "max": 15, "weight": 5}         # 5%
        }
        
        # 지속시간 범위 (분)
        self.duration_ranges = {
            "단기": {"min": 3, "max": 15, "weight": 35},       # 3-15분, 35%
            "중기": {"min": 16, "max": 45, "weight": 40},      # 16-45분, 40%
            "장기": {"min": 46, "max": 120, "weight": 20},     # 46분-2시간, 20%
            "초장기": {"min": 121, "max": 200, "weight": 5}    # 2시간 이상, 5%
        }
        
        # 장소명 - 쓰러짐/정지 상황이 발생할 수 있는 다양한 장소
        self.locations = [
            # 산업시설
            "제철소", "화학공장", "자동차공장", "조선소", "석유화학공장", "공장", "건설현장", 
            "물류창고", "자동차정비소", "주유소",
            
            # 의료시설
            "병원", "종합병원", "재활병원", "요양병원", "한의원", "의료센터", "약국",
            
            # 교통시설
            "지하철", "버스터미널", "고속터미널", "공항", "기차역",
            
            # 교육시설
            "어린이집", "대학교", "중학교", "학교", "도서관",
            
            # 상업시설
            "대형마트", "마트", "백화점", "쇼핑몰", "편의점", "세탁소", "미용실", "이발소",
            "카페", "레스토랑", "식당", "서점", "노래방",
            
            # 문화/여가시설
            "박물관", "과학관", "전시장", "콘서트홀", "영화관", "놀이공원", "키즈카페",
            "스포츠센터", "수영장", "사우나", "온천",
            
            # 공공시설/장소
            "아파트", "경로당", "공원", "도심공원", "광장", "번화가", "지하상가", "지하도상가",
            "지하보도", "상업지구", "치안CCTV", "공중화장실",
            
            # 숙박시설
            "호텔", "펜션",
            
            # 금융시설
            "은행"
        ]
        
        # 장소별 전용 구역명 매핑
        self.location_specific_areas = {
            # 산업시설
            "제철소": ["용광로 2번 앞", "용광로 3번 앞", "타워크레인 하부 A구역", "드라이독 구역 3번 베이"],
            "화학공장": ["반응기 1호 옆 제어실", "증류탑 1번 옆"],
            "자동차공장": ["도장라인", "포장라인", "컨베이어벨트 3번라인 A구역"],
            "조선소": ["드라이독 구역 3번 베이"],
            "공장": ["포장라인", "컨베이어벨트 3번라인 A구역"],
            "건설현장": ["타워크레인 하부 A구역"],
            "물류창고": ["하역장", "입구"],
            "자동차정비소": ["정비베이"],
            "주유소": ["세차장", "편의점 내부"],
            
            # 의료시설
            "병원": ["응급실 대기구역 3번 침상 옆", "CT실 대기구역 B코너", "외래진료실 앞"],
            "종합병원": ["내과 대기실"],
            "재활병원": ["물리치료실 A동 2층", "물리치료실"],
            "요양병원": ["회복실"],
            "한의원": ["침실 2호", "침실"],
            "약국": ["상담실"],
            
            # 교통시설
            "지하철": ["5호선 환승통로 서편 에스컬레이터 앞", "7호선 승강장 중앙 대합실", 
                     "1호선 대합실 남쪽 끝", "9호선 환승통로 서편 계단", "환승센터 중앙홀"],
            "고속터미널": ["대합실 벤치 3번"],
            "공항": ["탑승게이트 A15 대기실", "출국장 면세구역 게이트 B12", "탑승구 C12"],
            "버스터미널": ["매표소 앞"],
            
            # 교육시설
            "어린이집": ["놀이실 매트존", "야외놀이터"],
            "대학교": ["도서관 3층 열람실 개인석 구역", "중앙도서관 1층"],
            "중학교": ["교실 2층 2-3반 뒷좌석"],
            
            # 상업시설
            "대형마트": ["푸드코트 중앙테이블 2번", "휴게공간", "고객센터 앞 대기구역", "계산대"],
            "마트": ["고객센터 앞 대기구역", "계산대"],
            "세탁소": ["앞 대기석", "앞"],
            "미용실": ["대기실 VIP석", "샴푸실"],
            "이발소": ["대기 의자"],
            "카페": ["실내석 창가 테이블", "테라스 야외 좌석"],
            "레스토랑": ["주방 조리대 옆 바닥", "주방 냉장고 앞"],
            "식당": ["홀 서빙대 앞 주방 입구"],
            "서점": ["아동도서 코너 읽기공간"],
            "노래방": ["복도"],
            
            # 문화/여가시설
            "박물관": ["상설전시관 입구 로비", "특별전시관"],
            "과학관": ["천체투영관"],
            "전시장": ["VR체험관"],
            "콘서트홀": ["VIP석 3층 B구역"],
            "영화관": ["로비 소파"],
            "놀이공원": ["회전목마 대기줄 앞쪽", "회전목마 대기줄"],
            "키즈카페": ["볼풀장"],
            "스포츠센터": ["요가실"],
            "수영장": ["풀사이드 라운지체어"],
            "사우나": ["휴게실 안마의자"],
            
            # 공공시설/장소
            "아파트": ["관리사무소 앞 민원실", "어린이 놀이터"],
            "경로당": ["마루 2층 휴게실"],
            "공원": ["산책로 벤치 7번", "야외무대 앞"],
            "도심공원": ["연못가 북쪽"],
            "광장": ["분수대 주변 북쪽"],
            "번화가": ["중앙광장 분수대 북쪽", "지하보도"],
            "지하상가": ["중앙통로 패션구역"],
            "지하도상가": ["패션구역"],
            "지하보도": ["중간 지점"],
            "상업지구": ["A구역"],
            "치안CCTV": ["12지점 상업지구 A구역"],
            "공중화장실": ["앞 벤치"],
            
            # 숙박시설
            "호텔": ["연회장 입구"],
            "펜션": ["야외 수영장", "로비 소파"],
            
            # 금융시설
            "은행": ["ATM 대기줄 3번 코너", "창구 대기석"]
        }
        
        # 인원 유형 (직업/신분)
        self.person_types = {
            "산업": ["작업자", "근로자", "직원"],
            "의료": ["환자", "승무원"],
            "일반": ["시민", "고객", "관람객", "관광객", "승객", "관객", "이용객", "투숙객", "손님"],
            "교육": ["아동", "학생", "어르신"]
        }
        
        # 자세/상태 표현
        self.postures = {
            "위험": [
                "쓰러진 자세", "쓰러진 상태", "쓰러진 채로", "엎드려 있는 상태", 
                "바닥에 누워 있음", "누워 있는 모습", "바닥에 누워 있는 모습"
            ],
            "휴식": [
                "원형으로 앉아 있음", "의자에 기댄 자세", "벽에 기대어 서 있는 상태",
                "바닥에 앉아 있음", "똑바로 앉은 자세", "서로 기대어 앉아 있는 상태",
                "벤치에 누워 있는 모습", "소파에 앉아 있음", "의자에 앉은 채",
                "바닥에 앉은 자세", "의자에 앉아서 휴식 중", "테이블에 앉아 있음",
                "돗자리에 앉아 있음", "플라스틱 의자에 앉아 있는 상태"
            ],
            "대기": [
                "서서 대기", "똑바로 서 있는 자세", "서서 대기 중", "줄 서 있음",
                "서서 준비작업 중", "서서 안내판을 보고 있음"
            ],
            "활동": [
                "쪼그려 앉아 있는 상태", "침대에 누워 있음", "같은 자세", 
                "운동매트 위에 옆으로 누워 있음", "책상에 엎드려 있음",
                "쪼그려 앉은 자세", "요가매트에 누워 있는 자세",
                "눈을 감고 앉아 있는 상태", "모여 앉아 있는 상태"
            ],
            "정지": [
                "미동 없음", "움직임 없음", "벽에 기대어 앉은 자세",
                "의자에 기대어 앉아 있음", "똑바로 앉아 있는 상태"
            ]
        }
        
        # 상황 유형별 패턴
        self.situation_patterns = {
            "위험상황": {  # 기준 시간 초과 + 위험 자세
                "weight": 45,
                "posture_types": ["위험"],
                "exceed_ratio": (1.5, 10.0)  # 기준의 1.5배~10배 초과
            },
            "초과상황": {  # 기준 시간 초과 + 일반 자세
                "weight": 35,
                "posture_types": ["휴식", "대기", "활동", "정지"],
                "exceed_ratio": (1.1, 3.0)   # 기준의 1.1배~3배 초과
            },
            "정상상황": {  # 기준 시간 내
                "weight": 15,
                "posture_types": ["휴식", "대기", "활동"],
                "exceed_ratio": (0.3, 0.9)   # 기준의 30%~90%
            },
            "기준없음": {  # 기준 시간 미설정
                "weight": 5,
                "posture_types": ["위험", "휴식", "대기", "활동", "정지"],
                "exceed_ratio": None
            }
        }
        
        # 조치 표현
        self.action_expressions = {
            "위험상황": [
                "즉시 119 신고하시기 바랍니다",
                "즉시 119신고를 시행하시기 바랍니다", 
                "즉시 119 신고가 시급합니다",
                "즉시 119신고와 현장안전관리자 출동을 요청합니다",
                "즉시 119신고가 필요합니다",
                "즉시 현장출동과 응급처치를 시행하시기 바랍니다"
            ],
            "긴급확인": [
                "역무원 현장확인을 요청드립니다",
                "현장관리자 즉시 확인하시기 바랍니다",
                "안전확인을 부탁드립니다",
                "긴급점검이 필요합니다",
                "항공사 운항팀의 긴급점검이 필요합니다",
                "보안팀의 즉각적인 안전 확인과 근무 상황 점검이 필요합니다"
            ],
            "일반관리": [
                "의료진 일반 관찰 유지하면 됩니다",
                "보육교사 계속 지켜봐 주세요",
                "체육교사 일반 지도만 하시면 됩니다",
                "담임교사 일반 지도하세요",
                "놀이기구 운영진 일반 안전관리 하세요",
                "매장관리자 일반 업무지도 하세요",
                "일반 관찰만 유지하면 됩니다"
            ],
            "서비스제공": [
                "도슨트 안내서비스를 제공해주시기 바랍니다",
                "관리소 직원이 업무처리를 신속히 해주시기 바랍니다",
                "직원 추가서비스 제공하세요",
                "매장직원 서비스 진행하세요",
                "서비스 진행 상황 확인이 필요합니다",
                "업무처리 신속히 해주세요"
            ],
            "상황모니터링": [
                "보안순찰 강화하고 상황 모니터링 지속하세요",
                "상가관리사무소 안내요원 파견 권고합니다",
                "공원관리소 상황파악만 하세요",
                "계속 관찰하세요",
                "상황 모니터링을 지속하시기 바랍니다",
                "일반적인 관찰만 유지하면 됩니다"
            ]
        }
        
    def generate_person_count(self) -> int:
        """인원 수 생성"""
        weights = [self.person_ranges[key]["weight"] for key in self.person_ranges.keys()]
        selected_range = random.choices(list(self.person_ranges.keys()), weights=weights)[0]
        range_info = self.person_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_duration(self) -> int:
        """지속시간 생성 (분)"""
        weights = [self.duration_ranges[key]["weight"] for key in self.duration_ranges.keys()]
        selected_range = random.choices(list(self.duration_ranges.keys()), weights=weights)[0]
        range_info = self.duration_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_baseline_time(self, duration: int, situation_type: str) -> int:
        """기준 시간 생성"""
        if situation_type == "기준없음":
            return None
        elif situation_type == "위험상황":
            # 기준의 1.5배~10배 초과하도록 기준 설정
            exceed_min, exceed_max = self.situation_patterns["위험상황"]["exceed_ratio"]
            baseline = int(duration / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
        elif situation_type == "초과상황":
            # 기준의 1.1배~3배 초과하도록 기준 설정
            exceed_min, exceed_max = self.situation_patterns["초과상황"]["exceed_ratio"]
            baseline = int(duration / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
        else:  # 정상상황
            # 기준의 30%~90% 수준으로 현재 시간 설정
            exceed_min, exceed_max = self.situation_patterns["정상상황"]["exceed_ratio"]
            baseline = int(duration / random.uniform(exceed_min, exceed_max))
            return max(1, baseline)
    
    def select_situation_type(self) -> str:
        """상황 유형 선택"""
        weights = [self.situation_patterns[key]["weight"] for key in self.situation_patterns.keys()]
        return random.choices(list(self.situation_patterns.keys()), weights=weights)[0]
    
    def generate_person_type(self, location: str) -> str:
        """장소에 맞는 인원 유형 선택"""
        if location in ["제철소", "화학공장", "자동차공장", "조선소", "석油화학공장", "공장", "건설현장", "물류창고", "자동차정비소", "주유소"]:
            return random.choice(self.person_types["산업"])
        elif location in ["병원", "종합병원", "재활병원", "요양병원", "한의원", "의료센터", "약국"]:
            if random.random() < 0.7:
                return random.choice(self.person_types["의료"])
            else:
                return random.choice(self.person_types["일반"])
        elif location in ["어린이집", "대학교", "중학교", "학교", "도서관", "경로당"]:
            if "어린이집" in location:
                return "아동"
            elif "경로당" in location:
                return "어르신"  
            elif random.random() < 0.8:
                return "학생"
            else:
                return random.choice(self.person_types["일반"])
        else:
            return random.choice(self.person_types["일반"])
    
    def generate_posture(self, situation_type: str) -> str:
        """상황 유형에 맞는 자세 선택"""
        posture_types = self.situation_patterns[situation_type]["posture_types"]
        selected_type = random.choice(posture_types)
        return random.choice(self.postures[selected_type])
    
    def generate_time_expression(self) -> str:
        """시간 표현 생성"""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        formats = [
            f"{hour:02d}:{minute:02d}",  # 07:25
            f"오전 {hour if hour <= 12 else hour-12}시 {minute}분",  # 오전 9시 40분
            f"오후 {hour-12 if hour > 12 else hour}시 {minute}분",  # 오후 4시 15분  
            f"새벽 {hour}시 {minute}분",  # 새벽 5시 10분
            f"{hour}시 {minute}분"  # 13시 20분
        ]
        
        if hour < 6:
            return f"새벽 {hour}시 {minute}분"
        elif hour < 12:
            return f"오전 {hour}시 {minute}분"
        elif hour < 18:
            return f"오후 {hour-12 if hour > 12 else hour}시 {minute}분"
        else:
            if random.random() < 0.3:
                return f"저녁 {hour-12}시 {minute}분"
            else:
                return f"{hour:02d}:{minute:02d}"
    
    def generate_location_with_area(self) -> str:
        """장소 + 구역명 생성"""
        location = random.choice(self.locations)
        
        # 80% 확률로 구역명 추가
        if random.random() < 0.8 and location in self.location_specific_areas:
            area = random.choice(self.location_specific_areas[location])
            return f"{location} {area}"
        else:
            return location
    
    def format_duration(self, duration_minutes: int) -> str:
        """지속시간을 자연스러운 형식으로 변환"""
        if duration_minutes >= 60:
            hours = duration_minutes // 60
            minutes = duration_minutes % 60
            if minutes == 0:
                return f"{hours}시간"
            else:
                return f"{hours}시간 {minutes}분간"
        else:
            return f"{duration_minutes}분간"
    
    def generate_baseline_expression(self, baseline_time: int) -> str:
        """기준시간 표현 생성"""
        if baseline_time is None:
            expressions = ["기준 없음", "허용시간 없음", "임계치 없음", "기준 미설정", 
                         "허용치 없음", "기준값 없음", "안전기준 없음", "허용 기준 없음"]
            return random.choice(expressions)
        else:
            expressions = ["안전기준", "기준", "임계치", "허용시간", "기준치", "제한시간", 
                         "이용기준", "관찰기준", "기준값", "작업기준", "치료기준", "허용치",
                         "대기한계", "서비스기준", "업무기준", "검사대기시간", "표준대기시간",
                         "수업기준", "진료대기시간", "관람기준", "회복기준", "휴식기준"]
            expr = random.choice(expressions)
            if baseline_time >= 60:
                hours = baseline_time // 60
                minutes = baseline_time % 60
                if minutes == 0:
                    return f"{expr} {hours}시간"
                else:
                    return f"{expr} {hours}시간 {minutes}분"
            else:
                return f"{expr} {baseline_time}분"
    
    def generate_input_data(self) -> Dict:
        """Input 데이터 생성"""
        situation_type = self.select_situation_type()
        person_count = self.generate_person_count()
        duration = self.generate_duration()
        baseline_time = self.generate_baseline_time(duration, situation_type)
        time = self.generate_time_expression()
        location = self.generate_location_with_area()
        person_type = self.generate_person_type(location.split()[0])
        posture = self.generate_posture(situation_type)
        
        return {
            "situation_type": situation_type,
            "person_count": person_count,
            "duration": duration,
            "baseline_time": baseline_time,
            "time": time,
            "location": location,
            "person_type": person_type,
            "posture": posture
        }
    
    def generate_input_string(self, data: Dict) -> str:
        """Input 문자열 생성"""
        time = data["time"]
        location = data["location"]
        person_count = data["person_count"]
        person_type = data["person_type"]
        duration_str = self.format_duration(data["duration"])
        posture = data["posture"]
        baseline_str = self.generate_baseline_expression(data["baseline_time"])
        
        # 다양한 Input 형식 생성
        formats = [
            f'"{time} {location}에서, {person_type} {person_count}명 {duration_str} {posture}, {baseline_str}"',
            f'"{person_type} {person_count}명이 {location}에서 {time}부터 {duration_str} {posture}, {baseline_str}"',
            f'"{location}에서 {time} {person_type} {person_count}명이 {duration_str} {posture} 중, {baseline_str}"',
            f'"{time} {location} 앞에서 {person_type} {person_count}명 {duration_str} {posture}, {baseline_str}"',
            f'"{person_type} {person_count}명이 {location}에서 {time}부터 {duration_str} {posture}입니다, {baseline_str}"'
        ]
        
        return random.choice(formats)
    
    def generate_situation_analysis(self, data: Dict) -> str:
        """상황 분석 문장 생성"""
        duration = data["duration"]
        baseline_time = data["baseline_time"]
        duration_str = self.format_duration(duration)
        posture = data["posture"]
        situation_type = data["situation_type"]
        
        if situation_type == "기준없음":
            return f"{duration_str} {posture}로 기준 미설정이나 {'장시간 체류입니다' if duration > 30 else '상황 확인이 필요합니다'}"
        
        elif situation_type in ["위험상황", "초과상황"]:
            diff = duration - baseline_time
            baseline_str = self.generate_baseline_expression(baseline_time).split()[0]  # "안전기준"만 추출
            
            if situation_type == "위험상황":
                return f"{duration_str} {posture}로 {baseline_str} {baseline_time}분을 {diff}분 넘어선 {'극도위험' if diff > baseline_time * 3 else '위험'}상황입니다"
            else:
                return f"{duration_str} {posture}로 {baseline_str} {baseline_time}분을 {diff}분 {'초과한' if diff < 10 else '넘어선'} 상황입니다"
        
        else:  # 정상상황
            baseline_str = self.generate_baseline_expression(baseline_time).split()[0]
            return f"{duration_str} {posture}로 {baseline_str} {baseline_time}분 이내 {'정상범위' if duration < baseline_time * 0.7 else '정상상황'}입니다"
    
    def generate_location_time_person_summary(self, data: Dict) -> str:
        """장소+시간+인원 요약 문장 생성"""
        location = data["location"]
        time = data["time"] 
        person_count = data["person_count"]
        person_type = data["person_type"]
        posture = data["posture"]
        
        return f"{location}에서 {time}부터 {person_type} {person_count}명이 {posture}"
    
    def generate_action_recommendation(self, data: Dict) -> str:
        """조치 권고 문장 생성"""
        situation_type = data["situation_type"]
        posture = data["posture"]
        person_count = data["person_count"]
        
        if situation_type == "위험상황" or "쓰러진" in posture:
            if person_count >= 2:
                return "있어 " + random.choice(self.action_expressions["위험상황"])
            else:
                return "있으니 " + random.choice(self.action_expressions["위험상황"])
        elif situation_type == "초과상황":
            if person_count >= 4:
                return "하고 있으니 " + random.choice(self.action_expressions["긴급확인"])
            else:
                return "하니 " + random.choice(self.action_expressions["서비스제공"])
        elif situation_type == "정상상황":
            return "으로 " + random.choice(self.action_expressions["일반관리"])
        else:  # 기준없음
            return "으니 " + random.choice(self.action_expressions["상황모니터링"])
    
    def generate_output_content(self, data: Dict) -> str:
        """Output 내용 생성"""
        situation_analysis = self.generate_situation_analysis(data)
        location_summary = self.generate_location_time_person_summary(data)
        action = self.generate_action_recommendation(data)
        
        # 2문장 (60%) 또는 3문장 (40%)
        if random.random() < 0.6:
            # 2문장: 상황분석 + 장소요약+조치
            return f"{situation_analysis}. {location_summary}{action}."
        else:
            # 3문장: 상황분석 + 장소요약 + 조치
            middle_phrases = [
                "상태가 지속되고 있습니다", "상황입니다", "중입니다", 
                "로 있습니다", "하고 있습니다", "로 확인됩니다"
            ]
            middle = random.choice(middle_phrases)
            return f"{situation_analysis}. {location_summary}{middle}. {action.split('니 ')[1] if '니 ' in action else action.split('어 ')[1] if '어 ' in action else action.split('로 ')[1]}."
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        input_data = self.generate_input_data()
        input_str = self.generate_input_string(input_data)
        output_str = self.generate_output_content(input_data)
        domain = "쓰러짐 및 장기 정지 감지"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인2 데이터셋 {count}개 생성 중...")
        
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
        situation_types = {"위험상황": 0, "초과상황": 0, "정상상황": 0, "기준없음": 0}
        
        # 인원 규모 분포  
        person_ranges = {"개인": 0, "소수": 0, "중간": 0, "다수": 0}
        
        # 자세 유형 분포
        posture_types = {"위험": 0, "휴식": 0, "대기": 0, "활동": 0, "정지": 0}
        
        for input_str, output_str, domain in dataset:
            # 상황 유형 분류
            if "극도위험" in output_str or "위험상황" in output_str:
                situation_types["위험상황"] += 1
            elif "초과" in output_str or "넘어선" in output_str:
                situation_types["초과상황"] += 1
            elif "정상범위" in output_str or "정상상황" in output_str:
                situation_types["정상상황"] += 1
            else:
                situation_types["기준없음"] += 1
            
            # 인원 수 추출
            person_match = re.search(r'(\d+)명', input_str)
            if person_match:
                person_count = int(person_match.group(1))
                if person_count == 1:
                    person_ranges["개인"] += 1
                elif person_count <= 3:
                    person_ranges["소수"] += 1
                elif person_count <= 6:
                    person_ranges["중간"] += 1
                else:
                    person_ranges["다수"] += 1
            
            # 자세 유형 분류
            if "쓰러진" in input_str or "누워" in input_str:
                posture_types["위험"] += 1
            elif "앉아" in input_str or "기대어" in input_str:
                posture_types["휴식"] += 1
            elif "서서" in input_str or "서 있는" in input_str:
                posture_types["대기"] += 1
            elif "쪼그려" in input_str or "엎드려" in input_str:
                posture_types["활동"] += 1
            else:
                posture_types["정지"] += 1
        
        return {
            "total_count": total_count,
            "situation_types": {
                "위험상황": f"{situation_types['위험상황']} ({situation_types['위험상황']/total_count*100:.1f}%)",
                "초과상황": f"{situation_types['초과상황']} ({situation_types['초과상황']/total_count*100:.1f}%)",
                "정상상황": f"{situation_types['정상상황']} ({situation_types['정상상황']/total_count*100:.1f}%)",
                "기준없음": f"{situation_types['기준없음']} ({situation_types['기준없음']/total_count*100:.1f}%)"
            },
            "person_ranges": {
                "개인(1명)": f"{person_ranges['개인']} ({person_ranges['개인']/total_count*100:.1f}%)",
                "소수(2-3명)": f"{person_ranges['소수']} ({person_ranges['소수']/total_count*100:.1f}%)",
                "중간(4-6명)": f"{person_ranges['중간']} ({person_ranges['중간']/total_count*100:.1f}%)",
                "다수(7-15명)": f"{person_ranges['다수']} ({person_ranges['다수']/total_count*100:.1f}%)"
            },
            "posture_types": {
                "위험자세": f"{posture_types['위험']} ({posture_types['위험']/total_count*100:.1f}%)",
                "휴식자세": f"{posture_types['휴식']} ({posture_types['휴식']/total_count*100:.1f}%)",
                "대기자세": f"{posture_types['대기']} ({posture_types['대기']/total_count*100:.1f}%)",
                "활동자세": f"{posture_types['활동']} ({posture_types['활동']/total_count*100:.1f}%)",
                "정지자세": f"{posture_types['정지']} ({posture_types['정지']/total_count*100:.1f}%)"
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
    generator = Domain2FallDetectionGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 저장 경로 설정
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain2_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 데이터셋 검증 및 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인2 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n⚠️ 상황 유형 분포:")
    print(f"   위험상황: {validation_results['situation_types']['위험상황']}")
    print(f"   초과상황: {validation_results['situation_types']['초과상황']}")
    print(f"   정상상황: {validation_results['situation_types']['정상상황']}")
    print(f"   기준없음: {validation_results['situation_types']['기준없음']}")
    print(f"\n👥 인원 규모 분포:")
    print(f"   개인(1명): {validation_results['person_ranges']['개인(1명)']}")
    print(f"   소수(2-3명): {validation_results['person_ranges']['소수(2-3명)']}")
    print(f"   중간(4-6명): {validation_results['person_ranges']['중간(4-6명)']}")
    print(f"   다수(7-15명): {validation_results['person_ranges']['다수(7-15명)']}")
    print(f"\n🤸 자세 유형 분포:")
    print(f"   위험자세: {validation_results['posture_types']['위험자세']}")
    print(f"   휴식자세: {validation_results['posture_types']['휴식자세']}")
    print(f"   대기자세: {validation_results['posture_types']['대기자세']}")
    print(f"   활동자세: {validation_results['posture_types']['활동자세']}")
    print(f"   정지자세: {validation_results['posture_types']['정지자세']}") 