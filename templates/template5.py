import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain5AccessControlGenerator:
    """
    도메인5 폐쇄시간 무단 출입 감지 데이터셋 생성기
    
    주요 기능:
    - 다양한 시설의 폐쇄시간 출입 상황 데이터 생성
    - 출입 시간과 기준 시간 비교 분석
    - 접근 주체별 적절한 조치 방안 제시
    - 야간/새벽 시간대 중심의 상황 처리
    - 자연스러운 2~3문장 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 출입 지속시간 범위 (초)
        self.access_duration_ranges = {
            "초단기": {"min": 1, "max": 2, "weight": 25},    # 1-2초, 25%
            "단기": {"min": 3, "max": 4, "weight": 40},      # 3-4초, 40%
            "중기": {"min": 5, "max": 6, "weight": 25},      # 5-6초, 25%
            "장기": {"min": 7, "max": 9, "weight": 10}       # 7-9초, 10%
        }
        
        # 기준 시간 범위 (초)
        self.baseline_duration_ranges = {
            "엄격": {"min": 1, "max": 2, "weight": 40},      # 1-2초, 40%
            "보통": {"min": 3, "max": 4, "weight": 45},      # 3-4초, 45%
            "관대": {"min": 5, "max": 6, "weight": 15}       # 5-6초, 15%
        }
        
        # 접근 주체 및 인원수
        self.access_subjects = {
            "개인_인물": {
                "subjects": ["외부인", "시민", "학생", "작업자", "방문자", "배송기사", "청소업체 직원", "수리업체 직원"],
                "count_range": (1, 1),
                "weight": 45
            },
            "소수_인물": {
                "subjects": ["외부인", "시민", "학생", "작업자", "방문자", "배송기사", "청소업체 직원", "수리업체 직원"],
                "count_range": (2, 3),
                "weight": 35
            },
            "개인_차량": {
                "subjects": ["트럭", "승용차", "배송차량", "승합차", "지게차"],
                "count_range": (1, 1),
                "weight": 15
            },
            "복수_차량": {
                "subjects": ["승용차", "이륜차", "배송차량"],
                "count_range": (2, 3),
                "weight": 5
            }
        }
        
        # 시설별 출입구역 매핑
        self.facility_areas = {
            # 교육시설
            "학교": ["후문", "체육관 출입구", "본관 1층 게이트", "도서관 게이트", "옥상 출입문"],
            "기숙사": ["서문", "북문", "옥상문", "후문"],
            "연구동": ["북문", "동문", "서문", "옥상 출입문"],
            
            # 산업시설
            "공장": ["A구역 정문", "C구역 정문", "후문 계단", "A구역 게이트"],
            "창고": ["B동 출입게이트", "A동 후문", "C동 출입구", "정문"],
            
            # 의료시설
            "병원": ["응급실 후문", "주차장 게이트", "본관 출입구", "주차장 입구"],
            
            # 주거시설
            "아파트": ["지하주차장 입구", "옥상문", "관리사무소 입구"],
            "오피스텔": ["옥상문", "지하주차장 게이트", "서문"],
            
            # 기타 시설
            "연구소": ["정문", "후문", "실험동 출입구"],
            "사무소": ["정문", "후문", "지하주차장"]
        }
        
        # 출입 행동 유형
        self.access_actions = [
            "진입", "출입", "접근", "정차", "체류", "대기", "문 개방", 
            "침입", "머무름", "출입 시도", "침입 시도"
        ]
        
        # 시간대별 설정 (폐쇄시간 중심)
        self.time_periods = {
            "새벽": {"hours": list(range(1, 7)), "weight": 35},      # 01:00-06:59, 35%
            "야간": {"hours": list(range(22, 24)), "weight": 30},    # 22:00-23:59, 30%
            "심야": {"hours": [0], "weight": 15},                    # 00:00-00:59, 15%
            "일반": {"hours": list(range(7, 22)), "weight": 20}      # 07:00-21:59, 20%
        }
        
        # 상황 유형별 설정
        self.situation_types = {
            "위험초과": {"weight": 45, "exceed_ratio": (1.5, 4.0)},    # 기준 1.5~4배 초과
            "경미초과": {"weight": 25, "exceed_ratio": (1.1, 1.5)},    # 기준 1.1~1.5배 초과
            "정상범위": {"weight": 20, "exceed_ratio": (0.5, 1.0)},    # 기준 내
            "기준없음": {"weight": 10, "exceed_ratio": None}           # 기준 미설정
        }
        
        # 시간 표현 형식
        self.time_formats = {
            "숫자형": {"weight": 30, "formats": ["HH:MM"]},               # 04:27 형식
            "한글형": {"weight": 70, "formats": ["한글시간"]}             # 새벽 3시 17분 형식
        }
        
        # 상황별 분석 표현
        self.situation_analysis = {
            "위험초과": [
                "기준 {baseline}초를 {diff}초 초과한 상황입니다",
                "기준 {baseline}초 대비 {diff}초 초과하여",
                "허용치 {baseline}초를 {diff}초 상회한 상황입니다"
            ],
            "경미초과": [
                "기준 {baseline}초를 {diff}초 초과한 상황입니다",
                "허용치 {baseline}초를 {diff}초 상회한 상황입니다"
            ],
            "정상범위": [
                "기준 {baseline}초 이내의 정상 범위입니다",
                "허용치 {baseline}초 범위 내로 정상 상황입니다",
                "기준치 이내로 문제없는 상황입니다",
                "정상 범위로 확인됩니다"
            ],
            "기준없음": [
                "기준이 없는 상황으로",
                "허용치가 미설정된 구간으로",
                "기준 미설정 상태로",
                "기준이 설정되지 않은 구간으로"
            ]
        }
        
        # 조치 방안 (상황별)
        self.action_responses = {
            "위험초과": [
                "관리자 현장 확인이 필요합니다", "보안팀 즉시 점검이 필요합니다",
                "관리사무소 즉시 상황 확인이 필요합니다", "긴급 점검이 필요합니다",
                "즉시 대응이 요구됩니다", "관리팀 즉시 대응이 필요합니다",
                "보안팀 긴급 대응이 필요합니다", "현장팀 점검을 실시해주세요"
            ],
            "경미초과": [
                "관리 체크가 필요합니다", "현장 담당자 확인을 부탁드립니다",
                "확인이 필요합니다", "담당팀 점검이 요구됩니다",
                "관리 확인이 필요합니다", "담당팀 검토가 필요합니다",
                "보안팀 검토가 필요합니다", "현장 확인이 필요합니다"
            ],
            "정상범위": [
                "지속적인 모니터링을 유지하겠습니다", "경계선 상황이므로 담당팀 확인을 권장합니다",
                "지속 관찰이 필요합니다", "주의 깊게 관찰하겠습니다",
                "출입 관리를 계속 유지하겠습니다", "지속적인 모니터링을 유지하겠습니다"
            ],
            "기준없음": [
                "현재는 상황 관찰만 진행하며 지속적 패턴 발생 시 담당팀 검토를 요청드립니다",
                "일시적 관찰을 지속합니다", "상황 관찰을 지속하겠습니다",
                "현재는 관찰 단계입니다", "단순 관찰만 지속하고 있습니다",
                "일반 관찰하고 있습니다", "상황만 파악하고 있습니다"
            ]
        }
        
    def generate_access_duration(self) -> int:
        """출입 지속시간 생성 (1~9초)"""
        weights = [self.access_duration_ranges[key]["weight"] for key in self.access_duration_ranges.keys()]
        selected_range = random.choices(list(self.access_duration_ranges.keys()), weights=weights)[0]
        
        range_info = self.access_duration_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_baseline_duration(self, access_duration: int, situation_type: str) -> int:
        """기준 시간 생성"""
        if situation_type == "기준없음":
            return None
        
        if situation_type == "정상범위":
            # 출입시간보다 크거나 같게 설정 (정상 범위)
            return access_duration + random.randint(0, 2)
        else:
            # 출입시간보다 작게 설정 (초과 상황)
            exceed_ratio = self.situation_types[situation_type]["exceed_ratio"]
            if exceed_ratio:
                min_ratio, max_ratio = exceed_ratio
                baseline = int(access_duration / random.uniform(min_ratio, max_ratio))
                return max(1, baseline)
        
        # 기본값
        weights = [self.baseline_duration_ranges[key]["weight"] for key in self.baseline_duration_ranges.keys()]
        selected_range = random.choices(list(self.baseline_duration_ranges.keys()), weights=weights)[0]
        
        range_info = self.baseline_duration_ranges[selected_range]
        return random.randint(range_info["min"], range_info["max"])
    
    def generate_subject_info(self) -> Tuple[str, int]:
        """접근 주체 정보 생성"""
        weights = [self.access_subjects[key]["weight"] for key in self.access_subjects.keys()]
        selected_type = random.choices(list(self.access_subjects.keys()), weights=weights)[0]
        
        type_info = self.access_subjects[selected_type]
        subject = random.choice(type_info["subjects"])
        count = random.randint(*type_info["count_range"])
        
        return subject, count
    
    def generate_facility_info(self) -> Tuple[str, str]:
        """시설 및 출입구역 정보 생성"""
        facility = random.choice(list(self.facility_areas.keys()))
        
        if random.random() < 0.8:  # 80% 확률로 구체적 구역 추가
            area = random.choice(self.facility_areas[facility])
            return facility, area
        else:  # 20% 확률로 시설명만
            return facility, ""
    
    def generate_time_format(self) -> str:
        """시간 형식 생성 (폐쇄시간 중심)"""
        # 시간대 선택
        weights = [self.time_periods[key]["weight"] for key in self.time_periods.keys()]
        selected_period = random.choices(list(self.time_periods.keys()), weights=weights)[0]
        
        hour = random.choice(self.time_periods[selected_period]["hours"])
        minute = random.randint(0, 59)
        
        # 형식 선택
        format_type = random.choices(
            list(self.time_formats.keys()),
            weights=[self.time_formats[key]["weight"] for key in self.time_formats.keys()]
        )[0]
        
        if format_type == "숫자형":
            return f"{hour:02d}:{minute:02d}"
        else:  # 한글형
            if hour == 0:
                return f"자정 12시 {minute}분"
            elif 1 <= hour <= 6:
                return f"새벽 {hour}시 {minute}분"
            elif hour < 12:
                return f"오전 {hour}시 {minute}분"
            elif hour == 12:
                return f"정오 12시 {minute}분"
            elif hour < 18:
                return f"오후 {hour-12}시 {minute}분"
            elif hour < 22:
                return f"저녁 {hour-12}시 {minute}분"
            else:
                return f"밤 {hour-12}시 {minute}분"
    
    def select_situation_type(self) -> str:
        """상황 유형 선택"""
        weights = [self.situation_types[key]["weight"] for key in self.situation_types.keys()]
        return random.choices(list(self.situation_types.keys()), weights=weights)[0]
    
    def generate_input_string(self, data: Dict) -> str:
        """Input 문자열 생성"""
        time = data["time"]
        subject = data["subject"]
        count = data["count"]
        facility = data["facility"]
        area = data["area"]
        duration = data["duration"]
        baseline = data["baseline"]
        action = data["action"]
        
        # 주체 표현
        if count == 1:
            subject_expr = f"{subject} 1명" if subject in ["외부인", "시민", "학생", "작업자", "방문자", "배송기사", "청소업체 직원", "수리업체 직원"] else f"{subject} 1대"
        else:
            subject_expr = f"{subject} {count}명" if subject in ["외부인", "시민", "학생", "작업자", "방문자", "배송기사", "청소업체 직원", "수리업체 직원"] else f"{subject} {count}대"
        
        # 장소 표현
        if area:
            location_expr = f"{facility} {area}"
        else:
            location_expr = facility
        
        # 기준 시간 표현
        if baseline is None:
            baseline_expr = random.choice(["기준 없음", "기준 미설정", "허용치 없음", "허용치 미설정", "기준 정보 없음"])
        else:
            baseline_type = random.choice(["기준", "허용치"])
            baseline_expr = f"{baseline_type} {baseline}초"
        
        # Input 패턴 다양화
        patterns = [
            f"{time} {subject_expr} {location_expr} {duration}초 {action}, {baseline_expr}",
            f"{time}에 {subject_expr}이 {location_expr}에서 {duration}초간 {action}했습니다, {baseline_expr}",
            f"{duration}초 {action}한 {subject_expr} {time} {location_expr}, {baseline_expr}",
            f"{subject_expr}이 {time} {location_expr}에서 {duration}초간 {action}했습니다, {baseline_expr}"
        ]
        
        return random.choice(patterns)
    
    def generate_output_string(self, data: Dict) -> str:
        """Output 문자열 생성"""
        duration = data["duration"]
        baseline = data["baseline"]
        situation_type = data["situation_type"]
        time = data["time"]
        subject = data["subject"]
        count = data["count"]
        facility = data["facility"]
        area = data["area"]
        action = data["action"]
        
        # 주체 표현
        if count == 1:
            subject_expr = f"{subject} 1명" if subject in ["외부인", "시민", "학생", "작업자", "방문자", "배송기사", "청소업체 직원", "수리업체 직원"] else f"{subject} 1대"
        else:
            subject_expr = f"{subject} {count}명" if subject in ["외부인", "시민", "학생", "작업자", "방문자", "배송기사", "청소업체 직원", "수리업체 직원"] else f"{subject} {count}대"
        
        # 장소 표현
        if area:
            location_expr = f"{facility} {area}"
        else:
            location_expr = facility
        
        # 상황 분석 문장
        if situation_type == "기준없음":
            analysis = random.choice(self.situation_analysis[situation_type])
            situation = f"{time} {location_expr}에서 {subject_expr}이 {duration}초간 {action}했습니다"
        else:
            diff = duration - baseline
            if diff > 0:
                analysis = random.choice(self.situation_analysis[situation_type]).format(baseline=baseline, diff=diff)
                situation = f"{time} {location_expr}에서 {subject_expr}이 {duration}초간 {action}했습니다"
            else:
                analysis = random.choice(self.situation_analysis["정상범위"]).format(baseline=baseline)
                situation = f"{time} {location_expr}에서 {subject_expr}이 {duration}초간 {action}했습니다"
                situation_type = "정상범위"  # 분석 결과에 따라 상황 유형 조정
        
        # 조치 방안
        action_response = random.choice(self.action_responses[situation_type])
        
        # 2문장 패턴이 주류 (90%)
        if random.random() < 0.9:
            if situation_type == "기준없음":
                return f"{location_expr}에서 {time} {subject_expr}이 {duration}초간 {action}했습니다. {analysis} {action_response}."
            else:
                return f"{situation}. {analysis}. {action_response}."
        else:
            # 3문장 패턴 (10%)
            middle_info = self.generate_middle_sentence(situation_type, time)
            return f"{situation}. {analysis}. {middle_info} {action_response}."
    
    def generate_middle_sentence(self, situation_type: str, time: str) -> str:
        """중간 문장 생성 (3문장 패턴용)"""
        if situation_type in ["위험초과"]:
            return random.choice([
                "보안 규정 위반사항이 발생했습니다.",
                "즉각적인 대응이 필요한 상황입니다.",
                "폐쇄시간 무단 출입이 감지되었습니다."
            ])
        elif situation_type == "경미초과":
            return random.choice([
                "기준 초과로 주의가 필요합니다.",
                "추가 관리 조치가 요구됩니다.",
                "출입 통제 강화가 필요합니다."
            ])
        elif situation_type == "정상범위":
            if "새벽" in time or "밤" in time:
                return random.choice([
                    "야간 시간대 접근 모니터링을 지속하겠습니다.",
                    "새벽 시간대 접근 관리를 유지하겠습니다.",
                    "야간 배송 업무로 판단됩니다."
                ])
            else:
                return "정상적인 업무 활동으로 간주됩니다."
        else:  # 기준없음
            return random.choice([
                "반복 발생 시 경비실 점검을 권장합니다.",
                "동일 패턴 반복 시 현장 점검을 실시하겠습니다.",
                "유사 접근 시 주의가 필요합니다."
            ])
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        # 상황 유형 선택
        situation_type = self.select_situation_type()
        
        # 기본 정보 생성
        duration = self.generate_access_duration()
        baseline = self.generate_baseline_duration(duration, situation_type)
        subject, count = self.generate_subject_info()
        facility, area = self.generate_facility_info()
        time = self.generate_time_format()
        action = random.choice(self.access_actions)
        
        # 데이터 딕셔너리
        data = {
            "time": time,
            "subject": subject,
            "count": count,
            "facility": facility,
            "area": area,
            "duration": duration,
            "baseline": baseline,
            "action": action,
            "situation_type": situation_type
        }
        
        # Input/Output 생성
        input_str = self.generate_input_string(data)
        output_str = self.generate_output_string(data)
        domain = "폐쇄시간 무단 출입 감지"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인5 데이터셋 {count}개 생성 중...")
        
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
        time_periods = {"새벽": 0, "야간": 0, "심야": 0, "일반": 0}
        subject_types = Counter()
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        duration_dist = {"1-2초": 0, "3-4초": 0, "5-6초": 0, "7-9초": 0}
        facility_types = Counter()
        
        for input_str, output_str, domain in dataset:
            # 시간 형식 체크
            if re.search(r'\d{2}:\d{2}', input_str):
                time_formats["숫자형"] += 1
            else:
                time_formats["한글형"] += 1
            
            # 시간대 체크
            if "새벽" in input_str:
                time_periods["새벽"] += 1
            elif "밤" in input_str:
                time_periods["야간"] += 1
            elif "자정" in input_str:
                time_periods["심야"] += 1
            else:
                time_periods["일반"] += 1
            
            # 접근 주체 체크
            for subject_group in self.access_subjects.values():
                for subject in subject_group["subjects"]:
                    if subject in input_str:
                        subject_types[subject] += 1
                        break
            
            # 시설 유형 체크
            for facility in self.facility_areas.keys():
                if facility in input_str:
                    facility_types[facility] += 1
                    break
            
            # 지속시간 분포 체크
            duration_match = re.search(r'(\d+)초', input_str)
            if duration_match:
                duration = int(duration_match.group(1))
                if 1 <= duration <= 2:
                    duration_dist["1-2초"] += 1
                elif 3 <= duration <= 4:
                    duration_dist["3-4초"] += 1
                elif 5 <= duration <= 6:
                    duration_dist["5-6초"] += 1
                elif 7 <= duration <= 9:
                    duration_dist["7-9초"] += 1
            
            # 문장 길이 체크
            sentence_count = output_str.count('.')
            if sentence_count == 2:
                sentence_lengths["2문장"] += 1
            elif sentence_count == 3:
                sentence_lengths["3문장"] += 1
            else:
                sentence_lengths["기타"] += 1
            
            # 상황 유형 추정
            if "기준" not in input_str and "허용치" not in input_str:
                situation_types["기준없음"] += 1
            elif any(word in output_str for word in ["긴급", "즉시", "보안팀"]):
                situation_types["위험초과"] += 1
            elif any(word in output_str for word in ["초과", "상회"]):
                situation_types["경미초과"] += 1
            else:
                situation_types["정상범위"] += 1
        
        total_count = len(dataset)
        
        return {
            "total_count": total_count,
            "situation_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in situation_types.items()},
            "time_formats": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in time_formats.items()},
            "time_periods": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in time_periods.items()},
            "subject_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in subject_types.most_common(5)},
            "facility_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in facility_types.most_common(5)},
            "sentence_lengths": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in sentence_lengths.items()},
            "duration_dist": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in duration_dist.items()}
        }

if __name__ == "__main__":
    """메인 실행 부분"""
    generator = Domain5AccessControlGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 파일 저장
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain5_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 검증 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인5 데이터셋이 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n⚠️ 상황 유형 분포:")
    for k, v in validation_results['situation_types'].items():
        print(f"   {k}: {v}")
    print(f"\n⏰ 시간 형식 분포:")
    for k, v in validation_results['time_formats'].items():
        print(f"   {k}: {v}")
    print(f"\n🌙 시간대 분포:")
    for k, v in validation_results['time_periods'].items():
        print(f"   {k}: {v}")
    print(f"\n👥 접근 주체 분포 (상위 5개):")
    for k, v in validation_results['subject_types'].items():
        print(f"   {k}: {v}")
    print(f"\n🏢 시설 유형 분포 (상위 5개):")
    for k, v in validation_results['facility_types'].items():
        print(f"   {k}: {v}")
    print(f"\n📝 문장 길이 분포:")
    for k, v in validation_results['sentence_lengths'].items():
        print(f"   {k}: {v}")
    print(f"\n⏱️ 지속시간 분포:")
    for k, v in validation_results['duration_dist'].items():
        print(f"   {k}: {v}") 