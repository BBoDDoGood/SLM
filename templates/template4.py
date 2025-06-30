import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain4AccessDetectionGenerator:
    """
    도메인4 위험구역 접근 행위 감지 데이터셋 생성기
    
    주요 기능:
    - 다양한 장소의 위험구역 접근 상황 데이터 생성
    - 접근 시간과 기준 시간 비교 분석
    - 접근 주체별 적절한 조치 방안 제시
    - 상황별 위험도 판정 및 대응
    - 자연스러운 2~3문장 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 접근 시간 범위 (초)
        self.access_time_ranges = {
            "초단기": {"min": 1, "max": 2, "weight": 30},    # 1-2초, 30%
            "단기": {"min": 3, "max": 5, "weight": 40},      # 3-5초, 40%
            "중기": {"min": 6, "max": 8, "weight": 25},      # 6-8초, 25%
            "장기": {"min": 9, "max": 10, "weight": 5}       # 9-10초, 5%
        }
        
        # 기준 시간 범위 (초)
        self.baseline_time_ranges = {
            "엄격": {"min": 1, "max": 2, "weight": 35},      # 1-2초, 35%
            "보통": {"min": 3, "max": 4, "weight": 45},      # 3-4초, 45%
            "관대": {"min": 5, "max": 6, "weight": 20}       # 5-6초, 20%
        }
        
        # 접근 주체 (인원수 포함)
        self.access_subjects = {
            "개인": {
                "subjects": ["근로자", "학생", "시민", "외부인", "아동"],
                "count_range": (1, 1),
                "weight": 50
            },
            "소수": {
                "subjects": ["근로자", "학생", "시민", "외부인", "아동"],
                "count_range": (2, 3),
                "weight": 30
            },
            "다수": {
                "subjects": ["근로자", "학생", "시민", "외부인"],
                "count_range": (4, 5),
                "weight": 15
            },
            "장비": {
                "subjects": ["지게차", "트럭", "차량", "포크레인", "크레인"],
                "count_range": (1, 3),
                "weight": 5
            }
        }
        
        # 장소별 위험구역 매핑
        self.dangerous_locations = {
            # 산업시설
            "건설현장": ["크레인 하부", "타워크레인 작업반경", "굴착구역", "고압선로", "중장비 작업구역"],
            "제철소": ["용광로 냉각수 순환펌프실", "고로 냉각수 순환펌프실", "코크스 오븐 냉각탑 하부", "용광로 슬래그 처리구역"],
            "석유화학공장": ["반응기 안전구역", "폐수처리장 안전펜스", "수소가스 압축기실", "황산 저장탱크 배관"],
            "원자력발전소": ["냉각탑 차단구역", "사용후핵연료 저장수조", "핵연료 저장고", "방사선 차폐구역"],
            "화력발전소": ["보일러실 안전선", "석탄 저장고", "보일러 급수펌프실", "냉각수 취수구"],
            "조선소": ["용접작업장 경계선", "대형 크레인 회전반경", "선체 용접구역", "용접 스파크 비산구역"],
            "정유공장": ["원유탱크 보호구역", "휘발성가스 누출 감지선", "수소가스 압축기실"],
            "광산": ["갱도 입구 가스 감지구역", "갱도 메탄가스 감지구역"],
            
            # 교통시설
            "지하철": ["선로", "고압전선 보호구역", "변전실 안전선", "제3궤조 고압전류 보호덮개", "환기구 보호펜스"],
            "공항": ["활주로 안전구역", "항공기 급유시설", "연료저장고 보안펜스"],
            "항만": ["크레인 작업반경", "선박 연료공급라인", "컨테이너 적재구역", "컨테이너 크레인 하부"],
            "고속도로": ["공사구간 차선변경 차단봉", "터널 비상차선 진입금지선"],
            
            # 의료시설
            "병원": ["산소탱크 저장실", "방사선치료실 차폐벽", "방사선 폐기물 저장실", "MRI실 자기장 경고구역"],
            "대학병원": ["방사선치료실 차폐벽", "핵의학과 방사선 차폐실"],
            
            # 교육시설
            "학교": ["옥상", "화학실험실 후드 배기구", "과학실 화학폐기물"],
            "대학교": ["실험실 방사선 경고구역", "원자로 실험실 차폐문", "화학실험실 독성화학물질"],
            
            # 상업시설
            "냉동창고": ["암모니아 누출 구역", "액화질소 저장탱크"],
            "식품공장": ["냉동고 자동문", "무균실 에어락 차단선"],
            "대형마트": ["전기실 출입금지구역"],
            "쇼핑센터": ["전기설비실 출입구", "비상발전기실"],
            
            # 공공시설
            "댐": ["수문 조작실", "방류구 경고선", "수력발전기 터빈실", "수력터빈 냉각수 취수구"],
            "발전소": ["변전실", "석탄 저장고"],
            "수력발전소": ["터빈실 출입금지선"],
            
            # 주거시설
            "아파트": ["옥상 위험라인", "승강기 기계실", "보일러실 통풍구", "지하 전력케이블"],
            "놀이터": ["공사현장 펜스", "지하 전력시설 맨홀", "지하 가스배관 점검구"],
            
            # 기타 시설
            "물류창고": ["경계선"],
            "주차장": ["차량 접근구역"],
            "지하상가": ["변전실 출입금지구역", "비상계단 출입구"],
            "수영장": ["기계실 출입문"],
            "가스충전소": ["저장탱크 보호구역"],
            "반도체공장": ["클린룸 에어샤워 입구"]
        }
        
        # 상황 유형별 설정
        self.situation_types = {
            "극도위험": {"weight": 10, "exceed_ratio": (3.0, 6.0)},      # 기준 3~6배 초과
            "위험상황": {"weight": 35, "exceed_ratio": (1.5, 3.0)},      # 기준 1.5~3배 초과
            "초과상황": {"weight": 30, "exceed_ratio": (1.1, 1.5)},      # 기준 1.1~1.5배 초과
            "정상상황": {"weight": 20, "exceed_ratio": (0.3, 1.0)},      # 기준 내
            "기준없음": {"weight": 5, "exceed_ratio": None}              # 기준 미설정
        }
        
        # 시간 표현 형식
        self.time_formats = {
            "숫자형": {"weight": 60, "formats": ["HH:MM"]},               # 14:23 형식
            "한글형": {"weight": 40, "formats": ["한글시간"]}             # 오전 10시 33분 형식
        }
        
        # 조치 표현 (상황별)
        self.action_expressions = {
            "극도위험": [
                "즉시 비상프로토콜을 가동하고", "긴급 대피명령을 발령하고", 
                "즉시 현장 전체를 통제하고", "긴급 상황 대응팀을 투입하고"
            ],
            "위험상황": [
                "즉시 안전조치를 실시하고", "긴급히 현장을 확인하고",
                "즉시 해당 구역을 통제하고", "즉각적인 대응을 실시하고"
            ],
            "초과상황": [
                "안전점검을 실시하고", "현장 확인을 실시하고",
                "추가 안전조치를 취하고", "주의 조치를 시행하고"
            ],
            "정상상황": [
                "일반 안전수칙을 준수하고", "기본 안전관리를 유지하고",
                "예방 차원의 점검을 하고", "정기적인 확인을 하고"
            ],
            "기준없음": [
                "상황을 모니터링하고", "추이를 관찰하고",
                "현황을 파악하고", "동향을 점검하고"
            ]
        }
        
        # 구체적 조치 방안
        self.specific_actions = {
            "극도위험": [
                "소방서에 즉시 신고해주세요", "전 구역 대피를 실시해주세요",
                "비상 의료팀을 대기시켜주세요"
            ],
            "위험상황": [
                "해당 인원을 안전구역으로 이동시켜주세요", "즉시 작업을 중단해주세요",
                "관련 설비를 점검해주세요", "안전관리자가 현장 확인해주세요"
            ],
            "초과상황": [
                "해당 인원에게 주의를 주세요", "안전교육을 실시해주세요",
                "출입통제를 강화해주세요", "추가 순찰을 실시해주세요"
            ],
            "정상상황": [
                "안전거리 유지를 안내해주세요", "안전수칙을 재확인해주세요",
                "정기 점검을 실시해주세요", "예방 차원의 관리를 해주세요"
            ],
            "기준없음": [
                "반복 접근 시 확인해주세요", "동일 상황 발생 시 점검해주세요",
                "추가 발생 시마다 확인해주세요", "유사 접근 시 주의해주세요"
            ]
        }
        
    def generate_access_time(self) -> int:
        """접근 시간 생성 (1~10초)"""
        weights = [self.access_time_ranges[key]["weight"] for key in self.access_time_ranges.keys()]
        selected_range = random.choices(list(self.access_time_ranges.keys()), weights=weights)[0]
        
        range_info = self.access_time_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_baseline_time(self, access_time: int, situation_type: str) -> int:
        """기준 시간 생성"""
        if situation_type == "기준없음":
            return None
        
        if situation_type == "정상상황":
            # 접근시간보다 크게 설정 (정상 범위)
            return access_time + random.randint(1, 3)
        else:
            # 접근시간보다 작게 설정 (위험 상황)
            exceed_ratio = self.situation_types[situation_type]["exceed_ratio"]
            if exceed_ratio:
                min_ratio, max_ratio = exceed_ratio
                baseline = int(access_time / random.uniform(min_ratio, max_ratio))
                return max(1, baseline)
        
        # 기본값
        weights = [self.baseline_time_ranges[key]["weight"] for key in self.baseline_time_ranges.keys()]
        selected_range = random.choices(list(self.baseline_time_ranges.keys()), weights=weights)[0]
        
        range_info = self.baseline_time_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_subject_info(self) -> Tuple[str, int]:
        """접근 주체 정보 생성"""
        weights = [self.access_subjects[key]["weight"] for key in self.access_subjects.keys()]
        selected_type = random.choices(list(self.access_subjects.keys()), weights=weights)[0]
        
        type_info = self.access_subjects[selected_type]
        subject = random.choice(type_info["subjects"])
        count = random.randint(*type_info["count_range"])
        
        return subject, count
    
    def generate_location_info(self) -> Tuple[str, str]:
        """장소 및 위험구역 정보 생성"""
        location = random.choice(list(self.dangerous_locations.keys()))
        
        if random.random() < 0.8:  # 80% 확률로 구체적 위험구역 추가
            danger_zone = random.choice(self.dangerous_locations[location])
            return location, danger_zone
        else:  # 20% 확률로 장소명만
            return location, ""
    
    def generate_time_format(self) -> str:
        """시간 형식 생성"""
        format_type = random.choices(
            list(self.time_formats.keys()),
            weights=[self.time_formats[key]["weight"] for key in self.time_formats.keys()]
        )[0]
        
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        if format_type == "숫자형":
            return f"{hour:02d}:{minute:02d}"
        else:  # 한글형
            if hour == 0:
                return f"자정 12시 {minute}분"
            elif hour < 12:
                return f"오전 {hour}시 {minute}분"
            elif hour == 12:
                return f"정오 12시 {minute}분"
            else:
                return f"오후 {hour-12}시 {minute}분"
    
    def select_situation_type(self) -> str:
        """상황 유형 선택"""
        weights = [self.situation_types[key]["weight"] for key in self.situation_types.keys()]
        return random.choices(list(self.situation_types.keys()), weights=weights)[0]
    
    def generate_input_string(self, data: Dict) -> str:
        """Input 문자열 생성"""
        time = data["time"]
        subject = data["subject"]
        count = data["count"]
        location = data["location"]
        danger_zone = data["danger_zone"]
        access_time = data["access_time"]
        baseline_time = data["baseline_time"]
        
        # 주체 표현
        if count == 1:
            subject_expr = f"{subject} 1명"
        else:
            subject_expr = f"{subject} {count}명"
        
        # 장소 표현
        if danger_zone:
            location_expr = f"{location} {danger_zone}"
        else:
            location_expr = location
        
        # 기준 시간 표현
        if baseline_time is None:
            baseline_expr = random.choice(["기준 없음", "기준 미설정", "허용치 없음", "임계값 없음", "경보설정 없음"])
        else:
            baseline_type = random.choice(["기준", "임계값", "허용치", "경보설정"])
            baseline_expr = f"{baseline_type} {baseline_time}초"
        
        # Input 패턴 다양화
        patterns = [
            f"{time} {subject_expr} {location_expr}, {access_time}초간 접근, {baseline_expr}",
            f"{subject_expr}이 {time} {location_expr}에서 {access_time}초간 접근했습니다, {baseline_expr}",
            f"{access_time}초간 접근 {subject_expr}, {baseline_expr}, {location_expr} {time}",
            f"{time}에 {subject_expr}이 {location_expr}를 {access_time}초간 침범했습니다, {baseline_expr}"
        ]
        
        return random.choice(patterns)
    
    def generate_output_string(self, data: Dict) -> str:
        """Output 문자열 생성"""
        access_time = data["access_time"]
        baseline_time = data["baseline_time"]
        situation_type = data["situation_type"]
        time = data["time"]
        subject = data["subject"]
        count = data["count"]
        location = data["location"]
        danger_zone = data["danger_zone"]
        
        # 상황 분석 문장
        if situation_type == "기준없음":
            if count == 1:
                situation = f"{time} {location}{'에서' if not danger_zone else ''} {danger_zone}에서 {subject} 1명의 {access_time}초간 접근이 감지되었습니다"
            else:
                situation = f"{time} {location}{'에서' if not danger_zone else ''} {danger_zone}에서 {subject} {count}명의 {access_time}초간 접근이 감지되었습니다"
        else:
            diff = access_time - baseline_time
            if diff > 0:
                if count == 1:
                    situation = f"{time} {location}{'에서' if not danger_zone else ''} {danger_zone}에서 {subject} 1명이 {access_time}초간 접근하여 기준 {baseline_time}초보다 {diff}초 길었습니다"
                else:
                    situation = f"{time} {location}{'에서' if not danger_zone else ''} {danger_zone}에서 {subject} {count}명이 {access_time}초간 접근하여 기준 {baseline_time}초보다 {diff}초 길었습니다"
            else:
                if count == 1:
                    situation = f"{time} {location}{'에서' if not danger_zone else ''} {danger_zone}에서 {subject} 1명이 {access_time}초간 접근했습니다"
                else:
                    situation = f"{time} {location}{'에서' if not danger_zone else ''} {danger_zone}에서 {subject} {count}명이 {access_time}초간 접근했습니다"
        
        # 조치 문장
        action_start = random.choice(self.action_expressions[situation_type])
        specific_action = random.choice(self.specific_actions[situation_type])
        
        # 2문장 또는 3문장 선택
        if random.random() < 0.6:  # 60% 확률로 2문장
            return f"{situation}. {action_start} {specific_action}."
        else:  # 40% 확률로 3문장
            middle_info = self.generate_middle_sentence(situation_type, location, danger_zone)
            return f"{situation}. {middle_info} {action_start} {specific_action}."
    
    def generate_middle_sentence(self, situation_type: str, location: str, danger_zone: str) -> str:
        """중간 문장 생성 (3문장 패턴용)"""
        if situation_type in ["극도위험", "위험상황"]:
            return random.choice([
                "안전사고 위험이 매우 높습니다.",
                "즉각적인 대응이 필요한 상황입니다.",
                "생명안전에 직결되는 위험상황입니다."
            ])
        elif situation_type == "초과상황":
            return random.choice([
                "안전규정 위반사항이 발생했습니다.",
                "기준 초과로 안전점검이 필요합니다.",
                "추가 안전조치가 요구됩니다."
            ])
        elif situation_type == "정상상황":
            return random.choice([
                "기준 범위 내의 정상적인 접근입니다.",
                "현재 안전한 수준에서 관리되고 있습니다.",
                "특별한 위험은 감지되지 않았습니다."
            ])
        else:  # 기준없음
            return random.choice([
                "기준이 미설정된 구역입니다.",
                "모니터링 상태를 유지하고 있습니다.",
                "추가 관찰이 필요한 상황입니다."
            ])
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        # 상황 유형 선택
        situation_type = self.select_situation_type()
        
        # 기본 정보 생성
        access_time = self.generate_access_time()
        baseline_time = self.generate_baseline_time(access_time, situation_type)
        subject, count = self.generate_subject_info()
        location, danger_zone = self.generate_location_info()
        time = self.generate_time_format()
        
        # 데이터 딕셔너리
        data = {
            "time": time,
            "subject": subject,
            "count": count,
            "location": location,
            "danger_zone": danger_zone,
            "access_time": access_time,
            "baseline_time": baseline_time,
            "situation_type": situation_type
        }
        
        # Input/Output 생성
        input_str = self.generate_input_string(data)
        output_str = self.generate_output_string(data)
        domain = "위험구역 접근 행위 감지"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인4 데이터셋 {count}개 생성 중...")
        
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
        situation_types = Counter()
        time_formats = {"숫자형": 0, "한글형": 0}
        subject_types = Counter()
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        access_time_dist = {"1-2초": 0, "3-5초": 0, "6-8초": 0, "9-10초": 0}
        
        for input_str, output_str, domain in dataset:
            # 시간 형식 체크
            if re.search(r'\d{2}:\d{2}', input_str):
                time_formats["숫자형"] += 1
            else:
                time_formats["한글형"] += 1
            
            # 접근 주체 체크
            for subject_group in self.access_subjects.values():
                for subject in subject_group["subjects"]:
                    if subject in input_str:
                        subject_types[subject] += 1
                        break
            
            # 접근 시간 분포 체크
            time_match = re.search(r'(\d+)초간', input_str)
            if time_match:
                access_time = int(time_match.group(1))
                if 1 <= access_time <= 2:
                    access_time_dist["1-2초"] += 1
                elif 3 <= access_time <= 5:
                    access_time_dist["3-5초"] += 1
                elif 6 <= access_time <= 8:
                    access_time_dist["6-8초"] += 1
                elif 9 <= access_time <= 10:
                    access_time_dist["9-10초"] += 1
            
            # 문장 길이 체크
            sentence_count = output_str.count('.')
            if sentence_count == 2:
                sentence_lengths["2문장"] += 1
            elif sentence_count == 3:
                sentence_lengths["3문장"] += 1
            else:
                sentence_lengths["기타"] += 1
            
            # 상황 유형 추정
            if "기준" not in input_str and "임계값" not in input_str:
                situation_types["기준없음"] += 1
            elif any(word in output_str for word in ["비상", "대피", "긴급"]):
                situation_types["극도위험"] += 1
            elif any(word in output_str for word in ["즉시", "안전조치", "현장"]):
                situation_types["위험상황"] += 1
            elif any(word in output_str for word in ["점검", "확인", "주의"]):
                situation_types["초과상황"] += 1
            else:
                situation_types["정상상황"] += 1
        
        total_count = len(dataset)
        
        return {
            "total_count": total_count,
            "situation_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in situation_types.items()},
            "time_formats": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in time_formats.items()},
            "subject_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in subject_types.most_common(5)},
            "sentence_lengths": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in sentence_lengths.items()},
            "access_time_dist": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in access_time_dist.items()}
        }

if __name__ == "__main__":
    """메인 실행 부분"""
    generator = Domain4AccessDetectionGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 파일 저장
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain4_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 검증 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인4 데이터셋이 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n⚠️ 상황 유형 분포:")
    for k, v in validation_results['situation_types'].items():
        print(f"   {k}: {v}")
    print(f"\n⏰ 시간 형식 분포:")
    for k, v in validation_results['time_formats'].items():
        print(f"   {k}: {v}")
    print(f"\n👥 접근 주체 분포 (상위 5개):")
    for k, v in validation_results['subject_types'].items():
        print(f"   {k}: {v}")
    print(f"\n📝 문장 길이 분포:")
    for k, v in validation_results['sentence_lengths'].items():
        print(f"   {k}: {v}")
    print(f"\n⏱️ 접근 시간 분포:")
    for k, v in validation_results['access_time_dist'].items():
        print(f"   {k}: {v}") 