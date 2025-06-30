import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain1CrowdDetectionGenerator:
    """
    도메인1 군중 밀집 및 체류 감지 데이터셋 생성기
    
    주요 기능:
    - 다양한 장소의 군중 밀집 상황 데이터 생성 (3~1000명)
    - 기준 인원과 현재 인원 비교 분석
    - 시간대별 상황 변화 예측
    - 안전 조치 및 관리 방안 제시
    - 자연스러운 2~3문장 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 인원 규모별 설정 (3~1000명 범위)
        self.person_ranges = {
            "소규모": {"min": 3, "max": 30, "weight": 30},     # 30%
            "중규모": {"min": 31, "max": 100, "weight": 45},   # 45%
            "대규모": {"min": 101, "max": 500, "weight": 20},  # 20%
            "초대규모": {"min": 501, "max": 1000, "weight": 5} # 5%
        }
        
        # 장소명 - 군중 밀집이 발생할 수 있는 다양한 장소
        self.locations = [
            # 교통시설
            "지하철역", "버스터미널", "공항", "기차역", "고속터미널", "항만터미널",
            
            # 상업시설  
            "백화점", "대형마트", "쇼핑몰", "아울렛", "복합쇼핑센터", "전통시장",
            "영화관", "매장", "상점가",
            
            # 문화/체육시설
            "콘서트홀", "야구장", "축구경기장", "농구경기장", "체육관", "경기장", 
            "박물관", "미술관", "전시관", "전시센터", "문화회관", "공연장", "극장",
            "축제장", "야외무대", "컨벤션센터",
            
            # 교육시설
            "대학교", "도서관", "학교", "강의실",
            
            # 의료시설
            "병원", "종합병원", "의료센터", "클리닉",
            
            # 공공시설
            "시청", "구청", "관공서", "법원", "우체국", "은행",
            "시민회관", "복합문화센터",
            
            # 레저시설
            "테마파크", "놀이공원", "워터파크", "스키장", "리조트",
            "월드컵경기장", "올림픽공원",
            
            # 기타 시설
            "호텔", "박람회장", "실내체육관"
        ]
        
        # 장소별 전용 구역명 매핑
        self.location_specific_areas = {
            # 교통시설
            "지하철역": ["출입구", "대합실", "플랫폼", "환승통로", "계단", "에스컬레이터"],
            "버스터미널": ["게이트", "대합실", "승차장", "하차장", "매표소", "대기실"],
            "공항": ["게이트", "출발게이트", "도착게이트", "보안검색대", "수하물찾는곳", "체크인카운터", 
                   "출국장", "입국장", "면세구역", "대기구역"],
            "기차역": ["승강장", "대합실", "매표소", "출입구", "대기실"],
            
            # 상업시설
            "백화점": ["매표소", "계산대", "푸드코트", "중앙광장", "아트리움", "에스컬레이터"],
            "쇼핑몰": ["매표소", "계산대", "푸드코트", "중앙광장", "아트리움", "에스컬레이터"],
            "영화관": ["매표소", "상영관", "로비", "매점"],
            
            # 문화/체육시설
            "콘서트홀": ["입구", "로비", "매표소", "관람석", "VIP라운지"],
            "야구장": ["출입구", "계단", "매표소", "관람석", "내야석", "외야석"],
            "축구경기장": ["출입구", "게이트", "매표소", "관람석"],
            "축구장": ["출입구", "게이트", "매표소", "관람석"],
            "박물관": ["입구", "전시실", "매표소", "로비"],
            "전시관": ["입구", "부스", "매표소", "로비"],
            "축제장": ["정문", "광장", "입구", "중앙무대", "외곽"],
            
            # 교육시설
            "도서관": ["열람실", "입구", "복도"],
            
            # 의료시설
            "병원": ["접수대", "복도", "대기실", "외래진료센터"],
            
            # 기타
            "터미널": ["게이트", "대기라인", "대기구역"],
            "실내체육관": ["로비", "관람석"]
        }
        
        # 공통 구역명
        self.common_areas = [
            "입구", "출입구", "로비", "대기실", "1층", "2층", "3층", "앞",
            "계단", "통로", "광장", "중앙", "매표소"
        ]
        
        # 상황 유형별 패턴
        self.situation_patterns = {
            "초과상황": {  # 기준 인원 초과
                "weight": 60,
                "risk_levels": ["높음", "중간", "낮음"]
            },
            "정상상황": {  # 기준 인원 내
                "weight": 30,
                "risk_levels": ["정상"]
            },
            "기준없음": {  # 기준 인원 미설정
                "weight": 10,
                "risk_levels": ["관찰필요"]
            }
        }
        
        # 시간 표현 패턴
        self.time_expressions = {
            "숫자형": {
                "patterns": ["{hour:02d}:{minute:02d}", "{hour}:{minute:02d}"],
                "weight": 60
            },
            "한글형": {
                "patterns": ["새벽 {hour}시 {minute}분", "오전 {hour}시 {minute}분", 
                           "오후 {hour}시 {minute}분", "{hour}시 {minute}분"],
                "weight": 40
            }
        }
        
        # 상황 분석 표현
        self.situation_expressions = {
            "초과_기본": [
                "기준 {baseline}명을 {diff}명 상회한 {current}명이",
                "적정 기준치 {baseline}명을 {diff}명 초과한 {current}명이",
                "권장 기준 {baseline}명을 {diff}명 넘어선 {current}명이",
                "설정 기준 {baseline}명을 {diff}명 웃도는 {current}명이",
                "허용 기준 {baseline}명을 {diff}명 뛰어넘은 {current}명이"
            ],
            "초과_대규모": [
                "기준 인원 {baseline}명을 {diff}명 초과한 {current}명이",
                "설정 기준 {baseline}명을 크게 초과한 {current}명이",
                "기준 인원 {baseline}명을 대폭 넘어선 {current}명이"
            ],
            "정상_범위": [
                "기준 인원 {baseline}명보다 {diff}명 적은 {current}명이",
                "허용 기준 {baseline}명보다 {diff}명 미달인 {current}명이",
                "운영 기준 {baseline}명보다 {diff}명 부족한 {current}명이",
                "안전 기준 {baseline}명보다 {diff}명 적은 {current}명이"
            ],
            "기준없음": [
                "기준 미설정 상태로 {current}명이",
                "기준값 미설정 상태에서 {current}명이",
                "허용치 없이 {current}명이",
                "임계값 미설정으로 {current}명이"
            ]
        }
        
        # 상황 동사 표현
        self.situation_verbs = [
            "집중되어 있습니다", "모여있습니다", "밀집되어 있습니다", "체류하고 있습니다",
            "머물고 있습니다", "대기하고 있습니다", "분포해 있습니다", "포착되었습니다",
            "집결해 있습니다", "몰려있습니다", "줄서고 있습니다", "대기 중입니다"
        ]
        
        # 예측/전망 표현
        self.prediction_expressions = [
            "추가 관중 유입으로 혼잡 가중이 우려됩니다",
            "더 많은 방문객 유입이 예상됩니다",
            "향후 인파 집중이 예상됩니다",
            "추가 인원 증가가 우려됩니다",
            "혼잡 심화가 우려됩니다",
            "더욱 혼잡할 것으로 예상됩니다",
            "인파 증가가 진행 중입니다",
            "승객 급증이 예상됩니다",
            "관람객 증가에 대비한 준비가 필요합니다"
        ]
        
        # 조치/관리 표현
        self.action_expressions = [
            "우회 경로 안내를 권장합니다",
            "추가 출입구 개방이 필요합니다",
            "안전사고 방지를 위해 즉시 우회 동선 안내가 필요합니다",
            "정렬 안내와 사전 번호표 발급을 권장합니다",
            "현재 운영 방식을 유지하면 됩니다",
            "인원 분산 조치와 추가 탑승 게이트 개방이 급선무입니다",
            "자동 발권기 이용 안내가 효과적입니다",
            "즉각적인 인원 분산 조치가 필수입니다",
            "추가 체크인 카운터 개방과 대기선 재배치가 시급합니다",
            "분산 유도가 필요합니다",
            "안전사고 방지를 위한 입장 통제와 대기선 재정비가 시급합니다",
            "주변 공간으로의 분산 유도가 필요합니다",
            "즉각적인 흐름 분산과 대안 경로 안내가 필수적입니다"
        ]
        
        # 정상 상황 표현
        self.normal_expressions = [
            "원활한 흐름이 유지되고 있어 정상 운영이 가능합니다",
            "안정적인 통행 흐름이 이루어지고 있어 현재 상황은 양호합니다",
            "적정 범위 내에서 운영되고 있으나",
            "원활한 매표 서비스가 진행되고 있으며",
            "여유있는 상황이 지속되고 있어 현재 운영 방식을 유지하면 됩니다",
            "안정적인 수준이며",
            "여유로운 상태로 원활한 출입이 가능하며"
        ]
        
    def generate_person_count(self) -> int:
        """인원 수 생성"""
        weights = [self.person_ranges[key]["weight"] for key in self.person_ranges.keys()]
        selected_range = random.choices(list(self.person_ranges.keys()), weights=weights)[0]
        range_info = self.person_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_baseline_count(self, current_count: int, situation_type: str) -> int:
        """기준 인원 생성"""
        if situation_type == "초과상황":
            # 현재 인원보다 적게 설정 (위험 상황) - 최소값 보장
            if current_count <= 30:
                return max(3, current_count - random.randint(1, 8))
            elif current_count <= 100:
                return max(5, current_count - random.randint(5, 25))  # 최대 감소량 25로 제한
            else:
                return max(10, current_count - random.randint(20, 100))  # 최대 감소량 100으로 제한
        elif situation_type == "정상상황":
            # 현재 인원보다 많게 설정 (정상 상황)
            if current_count <= 30:
                return current_count + random.randint(1, 10)
            elif current_count <= 100:
                return current_count + random.randint(5, 25)
            else:
                return current_count + random.randint(20, 100)
        else:  # 기준없음
            return None
    
    def generate_time_expression(self) -> str:
        """시간 표현 생성"""
        hour = random.randint(6, 23)
        minute = random.randint(0, 59)
        
        time_type = random.choices(
            list(self.time_expressions.keys()),
            weights=[self.time_expressions[key]["weight"] for key in self.time_expressions.keys()]
        )[0]
        
        if time_type == "숫자형":
            pattern = random.choice(self.time_expressions["숫자형"]["patterns"])
            return pattern.format(hour=hour, minute=minute)
        else:  # 한글형
            if hour < 6:
                prefix = "새벽"
            elif hour < 12:
                prefix = "오전"
            elif hour < 18:
                prefix = "오후"
                hour = hour if hour == 12 else hour - 12
            else:
                prefix = "저녁"
                hour = hour - 12
            
            if minute == 0:
                return f"{prefix} {hour}시"
            else:
                return f"{prefix} {hour}시 {minute}분"
    
    def generate_location_with_area(self) -> str:
        """장소 + 구역명 생성"""
        location = random.choice(self.locations)
        
        # 70% 확률로 구역명 추가
        if random.random() < 0.7:
            if location in self.location_specific_areas and random.random() < 0.8:
                area = random.choice(self.location_specific_areas[location])
            else:
                area = random.choice(self.common_areas)
            
            # 자연스러운 조합 생성
            connectors = ["", " ", "의 ", " "]
            connector = random.choice(connectors)
            
            if area in ["앞", "입구", "출입구"]:
                return f"{location} {area}"
            else:
                return f"{location}{connector}{area}"
        else:
            return location
    
    def select_situation_type(self) -> str:
        """상황 유형 선택"""
        weights = [self.situation_patterns[key]["weight"] for key in self.situation_patterns.keys()]
        return random.choices(list(self.situation_patterns.keys()), weights=weights)[0]
    
    def generate_input(self) -> Dict:
        """Input 데이터 생성"""
        situation_type = self.select_situation_type()
        current_count = self.generate_person_count()
        baseline_count = self.generate_baseline_count(current_count, situation_type)
        time = self.generate_time_expression()
        location = self.generate_location_with_area()
        
        return {
            "situation_type": situation_type,
            "current_count": current_count,
            "baseline_count": baseline_count,
            "time": time,
            "location": location
        }
    
    def generate_situation_analysis(self, input_data: Dict) -> str:
        """상황 분석 문장 생성"""
        situation_type = input_data["situation_type"]
        current = input_data["current_count"]
        baseline = input_data["baseline_count"]
        time = input_data["time"]
        location = input_data["location"]
        
        # 기본 틀 생성
        time_location = f"{time} {location}에서"
        
        if situation_type == "초과상황":
            diff = current - baseline
            if current >= 100:
                expr = random.choice(self.situation_expressions["초과_대규모"])
            else:
                expr = random.choice(self.situation_expressions["초과_기본"])
            
            situation = expr.format(current=current, baseline=baseline, diff=diff)
            verb = random.choice(self.situation_verbs)
            
            return f"{time_location} {situation} {verb}"
            
        elif situation_type == "정상상황":
            diff = baseline - current
            expr = random.choice(self.situation_expressions["정상_범위"])
            situation = expr.format(current=current, baseline=baseline, diff=diff)
            verb = random.choice(self.situation_verbs)
            
            return f"{time_location} {situation} {verb}"
            
        else:  # 기준없음
            expr = random.choice(self.situation_expressions["기준없음"])
            situation = expr.format(current=current)
            verb = random.choice(self.situation_verbs)
            
            return f"{time_location} {situation} {verb}"
    
    def generate_output_content(self, input_data: Dict) -> str:
        """Output 내용 생성"""
        situation_analysis = self.generate_situation_analysis(input_data)
        situation_type = input_data["situation_type"]
        
        # 문장 개수 결정 (2문장: 60%, 3문장: 40%)
        sentence_count = random.choices([2, 3], weights=[60, 40])[0]
        
        if sentence_count == 2:
            if situation_type == "초과상황":
                # 위험 상황 + 조치
                action = random.choice(self.action_expressions)
                return f"{situation_analysis}. {action}."
            elif situation_type == "정상상황":
                # 정상 상황 + 예측
                normal = random.choice(self.normal_expressions)
                prediction = random.choice(self.prediction_expressions)
                return f"{situation_analysis}. {normal} {prediction}."
            else:  # 기준없음
                # 현재 상황 + 예측
                prediction = random.choice(self.prediction_expressions)
                return f"{situation_analysis}. {prediction}."
        else:  # 3문장
            if situation_type == "초과상황":
                # 상황 + 이유 + 조치
                reasons = [
                    "경기 입장 시간이 임박하여 인파 증가가 진행 중이며",
                    "진입로 협착으로 대기열이 형성되고 있어",
                    "접수 개시 시간을 앞두고 대기인원이 증가하고 있어",
                    "안전사고 위험이 높아지고 있어",
                    "축제 메인 행사를 앞두고",
                    "점심시간 승객 유입으로 심각한 정체가 발생하고 있어"
                ]
                reason = random.choice(reasons)
                action = random.choice(self.action_expressions)
                prediction = random.choice(self.prediction_expressions)
                
                return f"{situation_analysis}. {reason} {prediction}. {action}."
            else:
                # 상황 + 정상표현 + 예측
                normal = random.choice(self.normal_expressions)
                prediction = random.choice(self.prediction_expressions)
                return f"{situation_analysis}. {normal}. {prediction}."
    
    def generate_input_string(self, input_data: Dict) -> str:
        """Input 문자열 생성"""
        time = input_data["time"]
        location = input_data["location"]
        current = input_data["current_count"]
        baseline = input_data["baseline_count"]
        
        # 다양한 Input 형식 생성
        formats = [
            f'"{time} {location} 앞 {current}명 계측, 기준 인원 {baseline}명"',
            f'"{location}에 {time} 현재 {current}명이 집계되고 있으며, 기준값은 {baseline}명으로 설정되어 있습니다."',
            f'"{time} {location}에서 {current}명 밀집, 기준 수용인원 {baseline}명"',
            f'"{time} {location} 앞 {current}명 파악, 기준 {baseline}명"',
            f'"{location}에서 {time} 현재 {current}명의 사람들이 모여있으며, 적정 기준치는 {baseline}명입니다."',
            f'"{location}에 {time} 현재 {current}명이 있으며, 운영 기준은 {baseline}명입니다."',
            f'"{time} {location} 감지 인원 {current}명, 기준 인원 {baseline}명"'
        ]
        
        if baseline is None:
            # 기준 없음 형식
            no_baseline_formats = [
                f'"{time} {location}에서 {current}명 파악, 기준값 없음입니다."',
                f'"{location}에 {time} 현재 {current}명이 있으며, 기준값 미설정입니다."',
                f'"{time} {location} 감지 인원 {current}명, 기준 미설정"',
                f'"{location}에 {time} 현재 {current}명이 있습니다. 임계값 없음입니다."'
            ]
            return random.choice(no_baseline_formats)
        else:
            return random.choice(formats)
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        input_data = self.generate_input()
        input_str = self.generate_input_string(input_data)
        output_str = self.generate_output_content(input_data)
        domain = "군중 밀집 및 체류 감지"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인1 데이터셋 {count}개 생성 중...")
        
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
        situation_types = {"초과상황": 0, "정상상황": 0, "기준없음": 0}
        
        # 인원 규모 분포
        person_ranges = {"소규모": 0, "중규모": 0, "대규모": 0, "초대규모": 0}
        
        # 시간 형식 분포
        time_formats = {"숫자형": 0, "한글형": 0}
        
        # 문장 길이 분포
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        
        for input_str, output_str, domain in dataset:
            # 상황 유형 분류
            if "초과" in output_str or "넘어" in output_str or "상회" in output_str:
                situation_types["초과상황"] += 1
            elif "적은" in output_str or "미달" in output_str or "부족한" in output_str:
                situation_types["정상상황"] += 1
            else:
                situation_types["기준없음"] += 1
            
            # 인원 수 추출 및 분류
            person_match = re.search(r'(\d+)명', input_str)
            if person_match:
                person_count = int(person_match.group(1))
                if person_count <= 30:
                    person_ranges["소규모"] += 1
                elif person_count <= 100:
                    person_ranges["중규모"] += 1
                elif person_count <= 500:
                    person_ranges["대규모"] += 1
                else:
                    person_ranges["초대규모"] += 1
            
            # 시간 형식 체크
            if re.search(r'\d{1,2}:\d{2}', input_str):
                time_formats["숫자형"] += 1
            else:
                time_formats["한글형"] += 1
            
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
        
        return {
            "total_count": total_count,
            "situation_types": {
                "초과상황": f"{situation_types['초과상황']} ({situation_types['초과상황']/total_count*100:.1f}%)",
                "정상상황": f"{situation_types['정상상황']} ({situation_types['정상상황']/total_count*100:.1f}%)",
                "기준없음": f"{situation_types['기준없음']} ({situation_types['기준없음']/total_count*100:.1f}%)"
            },
            "person_ranges": {
                "소규모(3-30명)": f"{person_ranges['소규모']} ({person_ranges['소규모']/total_count*100:.1f}%)",
                "중규모(31-100명)": f"{person_ranges['중규모']} ({person_ranges['중규모']/total_count*100:.1f}%)",
                "대규모(101-500명)": f"{person_ranges['대규모']} ({person_ranges['대규모']/total_count*100:.1f}%)",
                "초대규모(501-1000명)": f"{person_ranges['초대규모']} ({person_ranges['초대규모']/total_count*100:.1f}%)"
            },
            "time_formats": {
                "숫자형": f"{time_formats['숫자형']} ({time_formats['숫자형']/total_count*100:.1f}%)",
                "한글형": f"{time_formats['한글형']} ({time_formats['한글형']/total_count*100:.1f}%)"
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
    generator = Domain1CrowdDetectionGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 저장 경로 설정
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain1_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 데이터셋 검증 및 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인1 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n📈 상황 유형 분포:")
    print(f"   초과상황: {validation_results['situation_types']['초과상황']}")
    print(f"   정상상황: {validation_results['situation_types']['정상상황']}")
    print(f"   기준없음: {validation_results['situation_types']['기준없음']}")
    print(f"\n👥 인원 규모 분포:")
    print(f"   소규모(3-30명): {validation_results['person_ranges']['소규모(3-30명)']}")
    print(f"   중규모(31-100명): {validation_results['person_ranges']['중규모(31-100명)']}")
    print(f"   대규모(101-500명): {validation_results['person_ranges']['대규모(101-500명)']}")
    print(f"   초대규모(501-1000명): {validation_results['person_ranges']['초대규모(501-1000명)']}")
    print(f"\n⏰ 시간 형식 분포:")
    print(f"   숫자형: {validation_results['time_formats']['숫자형']}")
    print(f"   한글형: {validation_results['time_formats']['한글형']}")
    print(f"\n📝 문장 길이 분포:")
    print(f"   2문장: {validation_results['sentence_lengths']['2문장']}")
    print(f"   3문장: {validation_results['sentence_lengths']['3문장']}")
    print(f"   기타: {validation_results['sentence_lengths']['기타']}")