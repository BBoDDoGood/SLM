import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain7TemperatureGenerator:
    """
    도메인7 온도 기반 쾌적도 및 폭염 예보 안내 데이터셋 생성기
    
    주요 기능:
    - 기온, 체감온도, 불쾌지수 기반 위험 관리 데이터 생성
    - 12가지 실외 장소별 온도 모니터링
    - 기준 온도/허용치 대비 위험도 분석
    - 지역별 온도 정보 및 예측 데이터
    - 2~3문장 자연스러운 흐름의 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 실외 장소 설정 (온도 모니터링 필요한 장소)
        self.locations = [
            "옥상", "야외 주차장", "자재 적치장", "컨테이너 주변", "창고 앞", 
            "공사장 입구", "마당", "운동장", "놀이터", "테라스", "건물 외부", "야외 작업장"
        ]
        
        # 전국 시/군/구 지역 설정
        self.regions = [
            # 서울특별시
            "서울특별시 강남구", "서울특별시 강동구", "서울특별시 강북구", "서울특별시 강서구",
            "서울특별시 관악구", "서울특별시 광진구", "서울특별시 구로구", "서울특별시 금천구",
            "서울특별시 노원구", "서울특별시 도봉구", "서울특별시 동대문구", "서울특별시 동작구",
            "서울특별시 마포구", "서울특별시 서대문구", "서울특별시 서초구", "서울특별시 성동구",
            "서울특별시 성북구", "서울특별시 송파구", "서울특별시 양천구", "서울특별시 영등포구",
            "서울특별시 용산구", "서울특별시 은평구", "서울특별시 종로구", "서울특별시 중구", "서울특별시 중랑구",
            
            # 부산광역시
            "부산광역시 강서구", "부산광역시 금정구", "부산광역시 기장군", "부산광역시 남구",
            "부산광역시 동구", "부산광역시 동래구", "부산광역시 부산진구", "부산광역시 북구",
            "부산광역시 사상구", "부산광역시 사하구", "부산광역시 서구", "부산광역시 수영구",
            "부산광역시 연제구", "부산광역시 영도구", "부산광역시 중구", "부산광역시 해운대구",
            
            # 대구광역시
            "대구광역시 남구", "대구광역시 달서구", "대구광역시 달성군", "대구광역시 동구",
            "대구광역시 북구", "대구광역시 서구", "대구광역시 수성구", "대구광역시 중구",
            
            # 인천광역시
            "인천광역시 강화군", "인천광역시 계양구", "인천광역시 남동구", "인천광역시 동구",
            "인천광역시 미추홀구", "인천광역시 부평구", "인천광역시 서구", "인천광역시 연수구",
            "인천광역시 옹진군", "인천광역시 중구",
            
            # 광주광역시
            "광주광역시 광산구", "광주광역시 남구", "광주광역시 동구", "광주광역시 북구", "광주광역시 서구",
            
            # 대전광역시
            "대전광역시 대덕구", "대전광역시 동구", "대전광역시 서구", "대전광역시 유성구", "대전광역시 중구",
            
            # 울산광역시
            "울산광역시 남구", "울산광역시 동구", "울산광역시 북구", "울산광역시 울주군", "울산광역시 중구",
            
            # 세종특별자치시
            "세종특별자치시",
            
            # 경기도 (주요 시/군)
            "경기도 고양시", "경기도 과천시", "경기도 광명시", "경기도 광주시", "경기도 구리시",
            "경기도 군포시", "경기도 김포시", "경기도 남양주시", "경기도 동두천시", "경기도 부천시",
            "경기도 성남시", "경기도 수원시", "경기도 시흥시", "경기도 안산시", "경기도 안성시",
            "경기도 안양시", "경기도 양주시", "경기도 오산시", "경기도 용인시", "경기도 의정부시",
            "경기도 이천시", "경기도 파주시", "경기도 평택시", "경기도 포천시", "경기도 하남시",
            "경기도 화성시", "경기도 가평군", "경기도 양평군", "경기도 연천군",
            
            # 강원도
            "강원도 강릉시", "강원도 동해시", "강원도 삼척시", "강원도 속초시", "강원도 원주시",
            "강원도 춘천시", "강원도 태백시", "강원도 고성군", "강원도 양구군", "강원도 양양군",
            "강원도 영월군", "강원도 인제군", "강원도 정선군", "강원도 철원군", "강원도 평창군",
            "강원도 홍천군", "강원도 화천군", "강원도 횡성군",
            
            # 충청북도
            "충청북도 제천시", "충청북도 청주시", "충청북도 충주시", "충청북도 괴산군", "충청북도 단양군",
            "충청북도 보은군", "충청북도 영동군", "충청북도 옥천군", "충청북도 음성군", "충청북도 증평군", "충청북도 진천군",
            
            # 충청남도
            "충청남도 계룡시", "충청남도 공주시", "충청남도 논산시", "충청남도 당진시", "충청남도 보령시",
            "충청남도 서산시", "충청남도 아산시", "충청남도 천안시", "충청남도 금산군", "충청남도 부여군",
            "충청남도 서천군", "충청남도 예산군", "충청남도 청양군", "충청남도 태안군", "충청남도 홍성군",
            
            # 전라북도
            "전라북도 군산시", "전라북도 김제시", "전라북도 남원시", "전라북도 익산시", "전라북도 전주시",
            "전라북도 정읍시", "전라북도 고창군", "전라북도 무주군", "전라북도 부안군", "전라북도 순창군",
            "전라북도 완주군", "전라북도 임실군", "전라북도 장수군", "전라북도 진안군",
            
            # 전라남도
            "전라남도 광양시", "전라남도 나주시", "전라남도 목포시", "전라남도 순천시", "전라남도 여수시",
            "전라남도 강진군", "전라남도 고흥군", "전라남도 곡성군", "전라남도 구례군", "전라남도 담양군",
            "전라남도 무안군", "전라남도 보성군", "전라남도 신안군", "전라남도 영광군", "전라남도 영암군",
            "전라남도 완도군", "전라남도 장성군", "전라남도 장흥군", "전라남도 진도군", "전라남도 함평군",
            "전라남도 해남군", "전라남도 화순군",
            
            # 경상북도
            "경상북도 경산시", "경상북도 경주시", "경상북도 구미시", "경상북도 김천시", "경상북도 문경시",
            "경상북도 상주시", "경상북도 안동시", "경상북도 영주시", "경상북도 영천시", "경상북도 포항시",
            "경상북도 고령군", "경상북도 군위군", "경상북도 봉화군", "경상북도 성주군", "경상북도 영덕군",
            "경상북도 영양군", "경상북도 예천군", "경상북도 울릉군", "경상북도 울진군", "경상북도 의성군",
            "경상북도 청도군", "경상북도 청송군", "경상북도 칠곡군",
            
            # 경상남도
            "경상남도 거제시", "경상남도 김해시", "경상남도 밀양시", "경상남도 사천시", "경상남도 양산시",
            "경상남도 진주시", "경상남도 창원시", "경상남도 통영시", "경상남도 거창군", "경상남도 고성군",
            "경상남도 남해군", "경상남도 산청군", "경상남도 의령군", "경상남도 창녕군", "경상남도 하동군",
            "경상남도 함안군", "경상남도 함양군", "경상남도 합천군",
            
            # 제주특별자치도
            "제주특별자치도 제주시", "제주특별자치도 서귀포시"
        ]
        
        # 온도 측정 유형별 설정
        self.temperature_types = {
            "기온": {"weight": 40, "range": (24, 36), "baseline": 30},
            "체감온도": {"weight": 35, "range": (24, 37), "baseline": 30},
            "불쾌지수": {"weight": 25, "range": (70, 87), "baseline": 75}
        }
        
        # 기준 설정 유형
        self.baseline_types = {
            "기준": {"weight": 70, "phrases": ["기준", "기준 온도", "기준값"]},
            "허용치": {"weight": 20, "phrases": ["허용치", "허용 기준", "기준치"]},
            "미설정": {"weight": 10, "phrases": ["기준 없음", "기준 미설정", "기준 없이"]}
        }
        
        # 시간대별 설정
        self.time_patterns = {
            "새벽": {"hours": range(4, 8), "weight": 15},
            "오전": {"hours": range(8, 12), "weight": 25},
            "낮": {"hours": range(12, 14), "weight": 20},
            "오후": {"hours": range(14, 18), "weight": 25},
            "저녁": {"hours": range(18, 22), "weight": 10},
            "야간": {"hours": range(22, 24), "weight": 5}
        }
        
        # 상황별 조치 표현
        self.response_patterns = {
            "극도위험": {
                "phrases": [
                    "즉시 중단하고 응급 냉각 조치를 시행하십시오",
                    "긴급 중단하고 응급 의료진을 즉시 배치해 주십시오",
                    "전면 중단하고 응급 냉각 시설을 가동해 주십시오",
                    "즉시 중단하고 냉방 시설로 대피하십시오"
                ],
                "weight": 10
            },
            "위험상황": {
                "phrases": [
                    "작업을 중단하고 냉방 시설이 있는 휴게소로 이동시키십시오",
                    "운영을 일시 중단하고 실내 대체 공간을 안내해 주십시오",
                    "활동을 즉시 중단하고 실내 냉방 공간으로 이동하십시오",
                    "사용을 중단하고 실내 놀이 공간을 안내해 주십시오"
                ],
                "weight": 25
            },
            "주의상황": {
                "phrases": [
                    "작업 시간을 단축하고 충분한 휴식을 확보해 주십시오",
                    "이용 시간을 제한하고 그늘막 설치를 검토하십시오",
                    "활동 시간을 단축하고 수분 보충을 강화하십시오",
                    "업무를 최소화하고 냉방 시설 가동을 검토하세요"
                ],
                "weight": 35
            },
            "정상관리": {
                "phrases": [
                    "현재 작업에 무리가 없으나 오전 중 작업 진행을 권장합니다",
                    "적합한 환경으로 이른 시간대 활용을 적극 권장합니다",
                    "최적의 조건이며 오전 중 활동을 적극 권장합니다",
                    "안전한 상태로 예방 조치를 준비해 주세요"
                ],
                "weight": 20
            },
            "지속모니터링": {
                "phrases": [
                    "지속적인 모니터링하겠습니다",
                    "계속 관찰해 주세요", 
                    "추가 모니터링을 강화해 주시기 바랍니다",
                    "상황을 면밀히 모니터링하십시오"
                ],
                "weight": 10
            }
        }
        
        # 예측 정보 표현
        self.prediction_patterns = [
            "내일 오전 {}시경에도 유사한 {}이 예상됩니다",
            "내일 오후 {}시경도 유사한 {}이 예상됩니다", 
            "오늘 오후 {}시경 추가 상승이 예상되므로",
            "오전 {}시 이후 급상승이 예상되므로",
            "내일 {}시경 상승이 예상되므로"
        ]
        
    def generate_time_format(self) -> str:
        """시간 형식 생성 (50% HH:MM, 50% 한글시간)"""
        hour = random.randint(4, 23)
        minute = random.randint(0, 59)
        
        if random.random() < 0.5:
            # HH:MM 형식
            return f"{hour:02d}:{minute:02d}"
        else:
            # 한글 시간 형식
            if hour < 6:
                return f"새벽 {hour}시 {minute:02d}분"
            elif hour < 12:
                return f"오전 {hour}시 {minute:02d}분"
            elif hour == 12:
                return f"낮 {hour}시 {minute:02d}분"
            elif hour < 18:
                return f"오후 {hour-12}시 {minute:02d}분"
            elif hour < 22:
                return f"저녁 {hour-12}시 {minute:02d}분"
            else:
                return f"야간 {hour-12}시 {minute:02d}분"
    
    def generate_temperature_data(self) -> Dict:
        """온도 데이터 생성"""
        # 온도 유형 선택
        temp_type = random.choices(
            list(self.temperature_types.keys()),
            weights=[self.temperature_types[t]["weight"] for t in self.temperature_types.keys()]
        )[0]
        
        config = self.temperature_types[temp_type]
        
        # 실제 온도값 생성
        actual_value = random.randint(config["range"][0], config["range"][1])
        
        # 기준값 생성 (실제값보다 낮게 설정하여 위험상황 시뮬레이션)
        if temp_type == "불쾌지수":
            if actual_value >= 80:
                baseline_value = random.randint(75, 80)
            else:
                baseline_value = random.randint(70, actual_value)
        else:  # 기온, 체감온도
            if actual_value >= 32:
                baseline_value = 30
            elif actual_value >= 30:
                baseline_value = random.choice([30, actual_value])
            else:
                baseline_value = 30
        
        # 기준 설정 유형 선택
        baseline_type = random.choices(
            list(self.baseline_types.keys()),
            weights=[self.baseline_types[t]["weight"] for t in self.baseline_types.keys()]
        )[0]
        
        return {
            "type": temp_type,
            "actual_value": actual_value,
            "baseline_value": baseline_value,
            "baseline_type": baseline_type,
            "difference": actual_value - baseline_value
        }
    
    def determine_risk_level(self, temp_data: Dict) -> str:
        """위험도 결정"""
        diff = temp_data["difference"]
        temp_type = temp_data["type"]
        
        if temp_data["baseline_type"] == "미설정":
            return "지속모니터링"
        
        if temp_type == "불쾌지수":
            if diff >= 6:
                return "극도위험"
            elif diff >= 4:
                return "위험상황"
            elif diff >= 1:
                return "주의상황"
            else:
                return "정상관리"
        else:  # 기온, 체감온도
            if diff >= 5:
                return "극도위험"
            elif diff >= 3:
                return "위험상황"
            elif diff >= 1:
                return "주의상황"
            else:
                return "정상관리"
    
    def generate_input(self) -> Dict:
        """Input 데이터 생성"""
        location = random.choice(self.locations)
        region = random.choice(self.regions)
        time = self.generate_time_format()
        temp_data = self.generate_temperature_data()
        
        return {
            "location": location,
            "region": region,
            "time": time,
            "temp_data": temp_data
        }
    
    def generate_input_string(self, input_data: Dict) -> str:
        """Input 문자열 생성"""
        location = input_data["location"]
        region = input_data["region"]
        time = input_data["time"]
        temp_data = input_data["temp_data"]
        
        # 다양한 Input 형식 패턴
        patterns = [
            # 패턴1: 시간 + 장소 + 지역 + 온도정보 + 기준정보
            f"{time} {location} {region} {temp_data['type']} {temp_data['actual_value']}도, {temp_data['baseline_type']} {temp_data['baseline_value']}도",
            
            # 패턴2: 장소 + 지역 + 시간 + 온도정보 + 기준정보
            f"{location} {region}에서 {time} {temp_data['type']}이 {temp_data['actual_value']}도로 {temp_data['baseline_type']} {temp_data['baseline_value']}도를 넘었습니다" if temp_data['difference'] > 0 else f"{location} {region}에서 {time} {temp_data['type']}이 {temp_data['actual_value']}도로 {temp_data['baseline_type']} {temp_data['baseline_value']}도 이하입니다",
            
            # 패턴3: 기준정보 먼저 + 나머지 정보
            f"{temp_data['baseline_type']} {temp_data['baseline_value']}도, {time} {location} {region} {temp_data['type']} {temp_data['actual_value']}도",
            
            # 패턴4: 온도정보 먼저 + 나머지 정보
            f"{temp_data['type']} {temp_data['actual_value']}도 {location} {region}에서 {time} {temp_data['baseline_type']} {temp_data['baseline_value']}도를 {temp_data['difference']}도 초과한 상황입니다" if temp_data['difference'] > 0 else f"{temp_data['type']} {temp_data['actual_value']}도 {location} {region}에서 {time} {temp_data['baseline_type']} {temp_data['baseline_value']}도 이하"
        ]
        
        return random.choice(patterns)
    
    def generate_output(self, input_data: Dict) -> str:
        """Output 생성 (2~3문장)"""
        location = input_data["location"]
        region = input_data["region"]
        time = input_data["time"]
        temp_data = input_data["temp_data"]
        
        risk_level = self.determine_risk_level(temp_data)
        
        # 첫 번째 문장: 상황 설명
        if temp_data["difference"] > 0:
            situation = f"{location} {region}에서 {time} {temp_data['type']}이 {temp_data['actual_value']}도로 확인되었습니다. {temp_data['baseline_type']} {temp_data['baseline_value']}도를 {temp_data['difference']}도 초과하여"
        elif temp_data["difference"] == 0:
            situation = f"{location} {region}에서 {time} {temp_data['type']}이 {temp_data['actual_value']}도로 나타났습니다. {temp_data['baseline_type']}과 일치하는 경계 상황으로"
        else:
            situation = f"{location} {region}에서 {time} {temp_data['type']}은 {temp_data['actual_value']}도로 나타났습니다. {temp_data['baseline_type']} {temp_data['baseline_value']}도 이하의 안전한 상태로"
        
        # 두 번째 문장: 위험 분석
        if risk_level == "극도위험":
            risk_analysis = "극도로 위험한 상황입니다"
        elif risk_level == "위험상황":
            risk_analysis = f"{location} 이용 시 열사병 위험이 큽니다"
        elif risk_level == "주의상황":
            risk_analysis = f"{location} 활동 시 주의가 필요합니다"
        elif risk_level == "정상관리":
            risk_analysis = f"{location} 활동에 최적의 환경입니다"
        else:
            risk_analysis = "계속 모니터링하겠습니다"
        
        # 세 번째 문장: 조치 방안
        action = random.choice(self.response_patterns[risk_level]["phrases"])
        
        # 예측 정보 추가 (30% 확률)
        prediction = ""
        if random.random() < 0.3 and risk_level in ["위험상황", "주의상황"]:
            pred_hour = random.randint(1, 4)
            if temp_data["type"] == "불쾌지수":
                pred_text = "위험 수준"
            else:
                pred_text = f"{temp_data['type']}"
            prediction = f" {random.choice(self.prediction_patterns).format(pred_hour, pred_text)}"
        
        # 문장 길이 결정 (2문장: 60%, 3문장: 40%)
        if random.random() < 0.6:
            # 2문장 패턴
            return f'"{situation} {risk_analysis}. {action}.{prediction}"'
        else:
            # 3문장 패턴
            return f'"{situation} {risk_analysis}. {action}.{prediction}"'
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        input_data = self.generate_input()
        input_str = self.generate_input_string(input_data)
        output_str = self.generate_output(input_data)
        domain = "온도 기반 쾌적도 및 폭염 예보 안내"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인7 데이터셋 {count}개 생성 중...")
        
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
        
        # 온도 유형 분포
        temp_types = {"기온": 0, "체감온도": 0, "불쾌지수": 0}
        # 위험도 분포  
        risk_levels = {"극도위험": 0, "위험상황": 0, "주의상황": 0, "정상관리": 0, "지속모니터링": 0}
        # 시간 형식 분포
        time_formats = {"HH:MM": 0, "한글시간": 0}
        
        for input_str, output_str, domain in dataset:
            # 온도 유형 체크
            if "기온" in input_str:
                temp_types["기온"] += 1
            elif "체감온도" in input_str:
                temp_types["체감온도"] += 1
            elif "불쾌지수" in input_str:
                temp_types["불쾌지수"] += 1
            
            # 위험도 체크
            if any(word in output_str for word in ["극도로", "즉시", "긴급"]):
                risk_levels["극도위험"] += 1
            elif any(word in output_str for word in ["중단하고", "일시 중단", "열사병"]):
                risk_levels["위험상황"] += 1
            elif any(word in output_str for word in ["주의", "단축", "제한"]):
                risk_levels["주의상황"] += 1
            elif any(word in output_str for word in ["최적", "권장", "무리가 없"]):
                risk_levels["정상관리"] += 1
            else:
                risk_levels["지속모니터링"] += 1
            
            # 시간 형식 체크
            if re.search(r'\d{2}:\d{2}', input_str):
                time_formats["HH:MM"] += 1
            else:
                time_formats["한글시간"] += 1
        
        return {
            "total_count": total_count,
            "temp_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in temp_types.items()},
            "risk_levels": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in risk_levels.items()},
            "time_formats": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in time_formats.items()}
        }

if __name__ == "__main__":
    # 데이터셋 생성기 인스턴스 생성
    generator = Domain7TemperatureGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 파일 저장
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain7_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 검증 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인7 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n🌡️ 온도 유형 분포:")
    for k, v in validation_results['temp_types'].items():
        print(f"   {k}: {v}")
    print(f"\n⚠️ 위험도 분포:")
    for k, v in validation_results['risk_levels'].items():
        print(f"   {k}: {v}")
    print(f"\n⏰ 시간 형식 분포:")
    for k, v in validation_results['time_formats'].items():
        print(f"   {k}: {v}") 