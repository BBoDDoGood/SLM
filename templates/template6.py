import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain6SafetyEquipmentGenerator:
    """
    도메인6 작업자 안전장비 미착용 감지 데이터셋 생성기
    
    주요 기능:
    - 다양한 작업장의 안전장비 미착용 상황 데이터 생성
    - 작업자별 미착용 장비와 기준 장비 비교 분석
    - 작업 유형별 위험도 평가 및 조치 방안 제시
    - 시간대별 작업 상황 처리
    - 자연스러운 2~3문장 Output 생성
    - 1000개 대용량 데이터셋 생성
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 작업자 인원수 설정
        self.worker_counts = {
            "개인": {"count": 1, "weight": 70},     # 1명, 70%
            "소수": {"count": 2, "weight": 25},     # 2명, 25%
            "다수": {"count": 3, "weight": 5}       # 3명, 5%
        }
        
        # 안전장비 유형 및 조합
        self.safety_equipment = {
            "마스크": {"risk": "분진 흡입", "action": "마스크 착용"},
            "헬멧": {"risk": "두부 손상", "action": "헬멧 착용"},
            "장갑": {"risk": "화학적 화상", "action": "장갑 착용"},
            "안전화": {"risk": "족부 외상", "action": "안전화 착용"},
            "안전복": {"risk": "화상", "action": "안전복 착용"},
            "보안경": {"risk": "시력 손상", "action": "보안경 착용"}
        }
        
        # 기준 장비 조합 (2개씩 조합)
        self.equipment_combinations = [
            ["마스크", "헬멧"], ["헬멧", "보안경"], ["안전화", "마스크"],
            ["장갑", "마스크"], ["헬멧", "안전복"], ["안전복", "마스크"],
            ["보안경", "마스크"], ["장갑", "헬멧"], ["안전복", "안전화"],
            ["보안경", "장갑"], ["마스크", "보안경"], ["헬멧", "안전화"],
            ["장갑", "보안경"], ["안전복", "보안경"], ["장갑", "안전화"]
        ]
        
        # 작업자 유형별 설정
        self.worker_types = {
            "산업_작업자": {
                "types": ["건설 기사", "용접공", "정비원", "철근공", "배관공", "전기 기사", "화학 작업자"],
                "weight": 40
            },
            "기술_관리자": {
                "types": ["점검원", "관리자", "기능공", "엔지니어", "검사원", "기술자"],
                "weight": 25
            },
            "현장_작업자": {
                "types": ["포장공", "설치원", "근로자", "청소원", "보수공", "운반원", "현장 작업자"],
                "weight": 20
            },
            "전문_기능공": {
                "types": ["타일공", "목수", "조립공", "시공자", "도장공"],
                "weight": 15
            }
        }
        
        # 작업장 유형별 구역 매핑
        self.workplace_areas = {
            "정비소": ["A구역", "B구역", "작업실", "도장실", "작업장"],
            "제조공장": ["A구역", "B블록", "C동", "품질실", "생산라인"],
            "건설현장": ["A동", "B구역", "C층", "D동", "E구역"],
            "공장": ["도장실", "용접실", "설비실", "A구역", "B동", "C블록"],
            "발전소": ["A동", "터빈실", "제어실", "발전기실"],
            "화학공장": ["정제동", "A동", "B동", "저장고", "연구동"],
            "조선소": ["조립장", "용접실", "도크", "건조장"]
        }
        
        # 작업 유형별 설정
        self.work_activities = {
            "제조_가공": {
                "activities": ["제조", "절단", "용접", "조립", "가공"],
                "high_risk": ["절단", "용접"],
                "weight": 25
            },
            "관리_점검": {
                "activities": ["관리", "점검", "정비", "보수", "검사"],
                "high_risk": ["정비", "보수"],
                "weight": 25
            },
            "운반_설치": {
                "activities": ["운반", "설치", "포장", "해체"],
                "high_risk": ["해체"],
                "weight": 20
            },
            "화학_전기": {
                "activities": ["화학물질 취급", "전기", "도장"],
                "high_risk": ["화학물질 취급", "전기"],
                "weight": 15
            },
            "건설_토목": {
                "activities": ["건설", "타일", "목공", "철근", "배관"],
                "high_risk": ["철근", "배관"],
                "weight": 10
            },
            "기타_작업": {
                "activities": ["청소", "기타"],
                "high_risk": [],
                "weight": 5
            }
        }
        
        # 미착용 표현 방식
        self.violation_expressions = {
            "미착용": ["{equipment} 미착용", "{equipment} 착용 없이", "{equipment} 없이"],
            "누락": ["{equipment} 착용 누락", "{equipment} 착용하지 않은 상태로", "{equipment} 빼고"],
            "부족": ["{equipment} 장비 없이", "{equipment} 장비 미비로", "{equipment} 보호구 없이"],
            "위반": ["{equipment} 착용 위반", "{equipment} 제대로 착용하지 않고", "{equipment} 벗은 채로"],
            "기타": ["{equipment} 안전장비 빠뜨리고", "{equipment} 보호구 빠뜨리고"]
        }
        
        # 시간 표현 형식
        self.time_formats = {
            "숫자형": {"weight": 35, "formats": ["HH:MM"]},        # 07:02, 14:23
            "한글형": {"weight": 65, "formats": ["한글시간"]}      # 오전 7시 2분, 새벽 3시
        }
        
        # 위험 상황별 표현
        self.risk_descriptions = {
            "분진 흡입": ["분진 흡입 위험", "유독가스 흡입 위험", "화학 분진 흡입 위험", "금속 분진 흡입 위험", "유해 분진 흡입"],
            "두부 손상": ["두부 손상 위험", "머리 부상 위험", "두부 외상 위험", "낙하물에 의한 두부 손상", "머리 외상"],
            "화학적 화상": ["화학적 화상 위험", "화학 물질 접촉 위험", "피부 손상 위험", "화학적 접촉", "도료 접촉"],
            "족부 외상": ["족부 외상 위험", "발 부상 위험", "족부 화상", "감전 시 발 부상", "발 손상"],
            "화상": ["화상 위험", "용접 불꽃에 의한 화상", "절단 불꽃에 의한 화상", "신체 손상", "접촉 화상"],
            "시력 손상": ["시력 손상 위험", "안구 손상 위험", "기계 파편에 의한 시력 손상", "화학 물질 비산", "용접 불꽃"]
        }
        
        # 작업 동사 표현 다양화 - 두 그룹으로 분리
        self.work_verbs_with_task = [
            "작업을 진행하고 있습니다", "작업을 수행하고 있습니다", "작업을 실시하고 있습니다", 
            "작업을 하고 있습니다", "작업 중입니다"
        ]
        
        self.work_verbs_without_task = [
            "진행하고 있습니다", "수행하고 있습니다", "실시하고 있습니다", 
            "하고 있습니다", "진행 중입니다"
        ]
        
        # 조치 표현 방식
        self.action_responses = {
            "즉시조치": {
                "phrases": [
                    "{action} 완료 후 {activity} 작업을 계속하세요",
                    "{action} 후 {activity} 작업을 계속하시기 바랍니다",
                    "{action} 완료 후 작업을 재개하세요",
                    "{action} 후 작업을 계속해 주십시오"
                ],
                "weight": 50
            },
            "점검요구": {
                "phrases": [
                    "{action} 상태를 점검하세요",
                    "{action} 상태 점검이 요구됩니다",
                    "{action} 확인이 필요합니다",
                    "{action} 개선이 시급합니다"
                ],
                "weight": 25
            },
            "지도실시": {
                "phrases": [
                    "{action} 지도를 실시하겠습니다",
                    "{action} 조치를 즉시 실시하겠습니다",
                    "{action} 조치를 신속히 진행해 주세요",
                    "즉시 대응이 요구됩니다"
                ],
                "weight": 15
            },
            "완료후재개": {
                "phrases": [
                    "{action} 완료 후 작업 재개가 필요합니다",
                    "{action} 완료 후 {activity} 작업을 지속하세요",
                    "{action} 후 작업 재개하시기 바랍니다"
                ],
                "weight": 10
            }
        }
        
    def generate_worker_count(self) -> int:
        """작업자 인원수 생성"""
        weights = [self.worker_counts[key]["weight"] for key in self.worker_counts.keys()]
        selected_type = random.choices(list(self.worker_counts.keys()), weights=weights)[0]
        return self.worker_counts[selected_type]["count"]
    
    def generate_worker_type(self) -> str:
        """작업자 유형 생성"""
        weights = [self.worker_types[key]["weight"] for key in self.worker_types.keys()]
        selected_category = random.choices(list(self.worker_types.keys()), weights=weights)[0]
        return random.choice(self.worker_types[selected_category]["types"])
    
    def generate_workplace_info(self) -> Tuple[str, str]:
        """작업장 및 구역 정보 생성"""
        workplace = random.choice(list(self.workplace_areas.keys()))
        
        if random.random() < 0.8:  # 80% 확률로 구체적 구역 추가
            area = random.choice(self.workplace_areas[workplace])
            return workplace, area
        else:  # 20% 확률로 작업장명만
            return workplace, ""
    
    def generate_work_activity(self) -> Tuple[str, bool]:
        """작업 활동 생성 (고위험 여부 포함)"""
        weights = [self.work_activities[key]["weight"] for key in self.work_activities.keys()]
        selected_category = random.choices(list(self.work_activities.keys()), weights=weights)[0]
        
        category_info = self.work_activities[selected_category]
        activity = random.choice(category_info["activities"])
        is_high_risk = activity in category_info["high_risk"]
        
        return activity, is_high_risk
    
    def generate_equipment_violation(self) -> Tuple[List[str], str, str]:
        """안전장비 위반 정보 생성"""
        # 기준 장비 조합 선택
        standard_equipment = random.choice(self.equipment_combinations)
        
        # 미착용 장비 선택 (기준 장비 중 1개)
        missing_equipment = random.choice(standard_equipment)
        
        # 미착용 표현 방식 선택
        violation_type = random.choice(list(self.violation_expressions.keys()))
        violation_expr = random.choice(self.violation_expressions[violation_type])
        violation_text = violation_expr.format(equipment=missing_equipment)
        
        return standard_equipment, missing_equipment, violation_text
    
    def generate_time_format(self) -> str:
        """시간 형식 생성"""
        hour = random.randint(0, 23)
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
            elif 1 <= hour <= 5:
                return f"새벽 {hour}시 {minute}분"
            elif 6 <= hour <= 11:
                return f"오전 {hour}시 {minute}분"
            elif hour == 12:
                return f"낮 12시 {minute}분"
            elif 13 <= hour <= 17:
                return f"오후 {hour-12}시 {minute}분"
            elif 18 <= hour <= 21:
                return f"저녁 {hour-12}시 {minute}분"
            else:  # 22-23
                return f"밤 {hour-12}시 {minute}분"
    
    def generate_input_string(self, data: Dict) -> str:
        """Input 문자열 생성"""
        time = data["time"]
        worker_type = data["worker_type"]
        worker_count = data["worker_count"]
        workplace = data["workplace"]
        area = data["area"]
        violation_text = data["violation_text"]
        activity = data["activity"]
        standard_equipment = data["standard_equipment"]
        
        # 작업자 표현
        worker_expr = f"{worker_type} {worker_count}명"
        
        # 장소 표현
        if area:
            location_expr = f"{workplace} {area}"
        else:
            location_expr = workplace
        
        # 기준 장비 표현
        standard_expr = "기준: " + ", ".join(standard_equipment)
        
        # Input 패턴 다양화 (domain6 예시와 동일한 패턴)
        patterns = [
            f"{location_expr} {time} {worker_expr} {violation_text} {activity}, {standard_expr}",
            f"{worker_expr} {time} {location_expr} {violation_text} {activity}, {standard_expr}",
            f"{standard_expr}, {time} {location_expr} {worker_expr} {violation_text} {activity}",
            f"{time} {location_expr}에서 {worker_expr}이 {violation_text} {activity}, {standard_expr}",
            f"{worker_expr} {location_expr} {time} {violation_text} {activity} 작업, {standard_expr}",
            f"{standard_expr}, {worker_expr} {time} {location_expr} {violation_text} {activity}"
        ]
        
        return random.choice(patterns).replace("이이", "이").replace("가가", "가")
    
    def generate_output_string(self, data: Dict) -> str:
        """Output 문자열 생성"""
        time = data["time"]
        worker_type = data["worker_type"]
        worker_count = data["worker_count"]
        workplace = data["workplace"]
        area = data["area"]
        missing_equipment = data["missing_equipment"]
        activity = data["activity"]
        standard_equipment = data["standard_equipment"]
        is_high_risk = data["is_high_risk"]
        
        # 작업자 표현
        worker_expr = f"{worker_type} {worker_count}명"
        
        # 장소 표현
        if area:
            location_expr = f"{workplace} {area}"
        else:
            location_expr = workplace
        
        # 동사 표현 선택 (50:50 비율)
        use_with_task = random.random() < 0.5
        if use_with_task:
            work_verb = random.choice(self.work_verbs_with_task)
            activity_expr = activity  # "작업을"이 포함된 표현이므로 활동명만
        else:
            work_verb = random.choice(self.work_verbs_without_task)
            activity_expr = f"{activity} 작업을"  # "작업을"을 추가
        
        # 상황 설명 문장
        if ":" in time:  # 숫자 시간
            situation = f"{location_expr}에서 {worker_expr}이 {missing_equipment} 없이 {activity_expr} {work_verb}"
        else:  # 한글 시간
            situation = f"{worker_expr}이 {location_expr}에서 {missing_equipment}을 착용하지 않고 {activity_expr} {work_verb}"
        
        # 위험 분석 문장
        equipment_risk = self.safety_equipment[missing_equipment]["risk"]
        risk_desc = random.choice(self.risk_descriptions[equipment_risk])
        
        other_equipment = [eq for eq in standard_equipment if eq != missing_equipment]
        other_eq_text = "과 ".join(other_equipment) if other_equipment else ""
        
        if other_equipment:
            risk_analysis = f"{missing_equipment} {'소홀히 하여' if not is_high_risk else '규정을 어겨'} {risk_desc}이 {'발생할 수 있습니다' if not is_high_risk else '매우 높습니다'}"
        else:
            risk_analysis = f"{missing_equipment} 미착용으로 {risk_desc}이 {'우려됩니다' if not is_high_risk else '매우 높습니다'}"
        
        # 조치 방안
        action_type = "즉시조치" if is_high_risk else random.choice(list(self.action_responses.keys()))
        action_template = random.choice(self.action_responses[action_type]["phrases"])
        action_text = self.safety_equipment[missing_equipment]["action"]
        
        if "{action}" in action_template and "{activity}" in action_template:
            action_response = action_template.format(action=action_text, activity=activity)
        elif "{action}" in action_template:
            action_response = action_template.format(action=action_text)
        else:
            action_response = action_template
        
        # 2문장 패턴이 주류 (85%)
        if random.random() < 0.85:
            return f"{situation}. {risk_analysis}. {action_response}."
        else:
            # 3문장 패턴 (15%) - 심각도에 따른 추가 정보
            if is_high_risk:
                additional_info = random.choice([
                    "안전장비 기준 위반이 감지되었습니다",
                    "심각한 안전장비 위반이 발견되었습니다",
                    f"안전장비 미착용 위험이 {location_expr}에서 발견되었습니다"
                ])
            else:
                additional_info = random.choice([
                    f"기준 위반 상황이 {location_expr}에서 발생했습니다",
                    f"{missing_equipment} 착용 누락이 확인됩니다",
                    "추가 안전 관리가 필요합니다"
                ])
            
            return f"{situation}. {risk_analysis}. {additional_info}. {action_response}."
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        # 기본 정보 생성
        worker_count = self.generate_worker_count()
        worker_type = self.generate_worker_type()
        workplace, area = self.generate_workplace_info()
        activity, is_high_risk = self.generate_work_activity()
        standard_equipment, missing_equipment, violation_text = self.generate_equipment_violation()
        time = self.generate_time_format()
        
        # 데이터 딕셔너리
        data = {
            "time": time,
            "worker_type": worker_type,
            "worker_count": worker_count,
            "workplace": workplace,
            "area": area,
            "violation_text": violation_text,
            "missing_equipment": missing_equipment,
            "activity": activity,
            "standard_equipment": standard_equipment,
            "is_high_risk": is_high_risk
        }
        
        # Input/Output 생성
        input_str = self.generate_input_string(data)
        output_str = self.generate_output_string(data)
        domain = "작업자 안전장비 미착용 감지"
        
        return input_str, output_str, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인6 데이터셋 {count}개 생성 중...")
        
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
        worker_counts = Counter()
        worker_types = Counter()
        workplaces = Counter()
        missing_equipment = Counter()
        activities = Counter()
        time_formats = {"숫자형": 0, "한글형": 0}
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        risk_levels = {"고위험": 0, "일반": 0}
        
        for input_str, output_str, domain in dataset:
            # 작업자 인원수 체크
            if "1명" in input_str:
                worker_counts["1명"] += 1
            elif "2명" in input_str:
                worker_counts["2명"] += 1
            elif "3명" in input_str:
                worker_counts["3명"] += 1
            
            # 시간 형식 체크
            if re.search(r'\d{2}:\d{2}', input_str):
                time_formats["숫자형"] += 1
            else:
                time_formats["한글형"] += 1
            
            # 작업자 유형 체크
            for category in self.worker_types.values():
                for worker_type in category["types"]:
                    if worker_type in input_str:
                        worker_types[worker_type] += 1
                        break
            
            # 작업장 체크
            for workplace in self.workplace_areas.keys():
                if workplace in input_str:
                    workplaces[workplace] += 1
                    break
            
            # 미착용 장비 체크
            for equipment in self.safety_equipment.keys():
                if equipment in input_str:
                    missing_equipment[equipment] += 1
                    break
            
            # 작업 활동 체크
            for category in self.work_activities.values():
                for activity in category["activities"]:
                    if activity in input_str:
                        activities[activity] += 1
                        break
            
            # 위험도 체크
            if any(word in output_str for word in ["매우 높습니다", "규정을 어겨", "심각한"]):
                risk_levels["고위험"] += 1
            else:
                risk_levels["일반"] += 1
            
            # 문장 길이 체크
            sentence_count = output_str.count('.')
            if sentence_count == 3:
                sentence_lengths["2문장"] += 1
            elif sentence_count == 4:
                sentence_lengths["3문장"] += 1
            else:
                sentence_lengths["기타"] += 1
        
        total_count = len(dataset)
        
        return {
            "total_count": total_count,
            "worker_counts": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in worker_counts.items()},
            "time_formats": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in time_formats.items()},
            "worker_types": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in worker_types.most_common(10)},
            "workplaces": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in workplaces.items()},
            "missing_equipment": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in missing_equipment.items()},
            "activities": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in activities.most_common(10)},
            "risk_levels": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in risk_levels.items()},
            "sentence_lengths": {k: f"{v} ({v/total_count*100:.1f}%)" for k, v in sentence_lengths.items()}
        }

if __name__ == "__main__":
    """메인 실행 부분"""
    generator = Domain6SafetyEquipmentGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 파일 저장
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain6_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 검증 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인6 데이터셋이 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    print(f"\n👥 작업자 인원 분포:")
    for k, v in validation_results['worker_counts'].items():
        print(f"   {k}: {v}")
    print(f"\n⏰ 시간 형식 분포:")
    for k, v in validation_results['time_formats'].items():
        print(f"   {k}: {v}")
    print(f"\n👷 작업자 유형 분포 (상위 10개):")
    for k, v in validation_results['worker_types'].items():
        print(f"   {k}: {v}")
    print(f"\n🏭 작업장 분포:")
    for k, v in validation_results['workplaces'].items():
        print(f"   {k}: {v}")
    print(f"\n⚠️ 미착용 장비 분포:")
    for k, v in validation_results['missing_equipment'].items():
        print(f"   {k}: {v}")
    print(f"\n🔧 작업 활동 분포 (상위 10개):")
    for k, v in validation_results['activities'].items():
        print(f"   {k}: {v}")
    print(f"\n⚡ 위험도 분포:")
    for k, v in validation_results['risk_levels'].items():
        print(f"   {k}: {v}")
    print(f"\n📝 문장 길이 분포:")
    for k, v in validation_results['sentence_lengths'].items():
        print(f"   {k}: {v}") 