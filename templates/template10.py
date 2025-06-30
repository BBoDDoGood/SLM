#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도메인10 이상 이동 패턴 감지 데이터셋 생성기
template8.py와 template9.py 구조를 기반으로 한 Domain10 전용 생성기

주요 기능:
- 7가지 이상 이동 패턴별 감지 데이터 생성 (동선 이탈, 궤적 반복, 비정상 방향 전환, 고속 이동, 급정지, 정지 후 재이동, 구역 경로 우회)
- 사람/차량 객체별 다양한 상황 시뮬레이션
- 거리/시간 기반 측정값과 기준값 비교
- 기준 설정/미설정 상황 처리
- 구조형/설명형 Input 생성 (60:40 비율)
- 2~4문장 자연스러운 Output 생성
- 1000개 대용량 데이터셋 생성
"""

import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain10AbnormalMovementGenerator:
    """
    도메인10 이상 이동 패턴 감지 데이터셋 생성기
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 이상 이동 패턴 유형별 설정 (7가지 주요 패턴)
        self.movement_patterns = {
            "동선 이탈": {"weight": 18, "unit_types": ["distance", "time"]},
            "궤적 반복": {"weight": 17, "unit_types": ["distance", "time"]},
            "비정상 방향 전환": {"weight": 16, "unit_types": ["distance", "time"]},
            "고속 이동": {"weight": 15, "unit_types": ["distance", "time"]},
            "급정지": {"weight": 14, "unit_types": ["distance", "time"]},
            "정지 후 재이동": {"weight": 12, "unit_types": ["distance", "time"]},
            "구역 경로 우회": {"weight": 8, "unit_types": ["distance", "time"]}
        }
        
        # 객체 유형 설정 (사람 + 차량)
        self.person_types = ["방문자", "직원", "외부인", "학생", "환자", "승객", "보행자"]
        self.vehicle_types = ["승용차", "트럭", "지게차"]
        
        # 측정 단위 설정
        self.measurement_units = {
            "distance": {
                "unit": "m",
                "formats": ["{value}m", "{value}m간"],  # 70% vs 30%
                "ranges": {
                    "짧음": {"min": 3, "max": 8, "weight": 30},
                    "보통": {"min": 9, "max": 15, "weight": 40},
                    "김": {"min": 16, "max": 20, "weight": 30}
                }
            },
            "time": {
                "unit": "초",
                "formats": ["{value}초", "{value}초간"],  # 70% vs 30%
                "ranges": {
                    "짧음": {"min": 3, "max": 7, "weight": 35},
                    "보통": {"min": 8, "max": 12, "weight": 40},
                    "김": {"min": 13, "max": 20, "weight": 25}
                }
            }
        }
        
        # 장소 설정 (실내·실외 다양한 실제 장소)
        self.locations = {
            # 단순 장소 (30-40%)
            "simple": [
                "물류센터", "공장", "사무실", "병원", "학교", "공항", "주차장", "운동장",
                "로비", "출국장", "입구", "출구", "복도", "계단", "엘리베이터"
            ],
            # 장소+구역명 (60-70%)
            "complex": [
                "물류센터 A구역", "물류센터 B구역", "공장 A동", "공장 B동", "공장 C동",
                "사무실 A구역", "사무실 B구역", "사무실 C구역", "병원 응급실 로비",
                "학교 운동장", "공항 출국장", "공항 입국장", "주차장 A구역", 
                "주차장 B구역", "주차장 C구역", "주차장 D구역", "A구역", "B구역", 
                "C구역", "D구역", "E구역", "F구역", "1층 로비", "2층 대기실",
                "지하1층", "옥상", "중앙광장", "휴게실", "회의실", "대기실"
            ]
        }
        
        # 기준 설정 유형
        self.baseline_types = {
            "기준있음": {
                "weight": 70,  # 기준 있음 70%
                "formats": ["기준: {value}{unit}", "기준 {value}{unit}"]
            },
            "기준없음": {
                "weight": 30,  # 기준 없음 30%
                "formats": ["허용치 없음", "기준 미설정", "기준 없음", "허용치 미설정"]
            }
        }
        
        # 위험도 분류 비율 (설계서 기준)
        self.risk_categories = {
            "위험": {"weight": 70, "threshold_ratio": 1.2},    # 기준 초과 시 70%
            "주의": {"weight": 20, "threshold_ratio": 0.9},    # 모니터링 강화 20%
            "정상": {"weight": 10, "threshold_ratio": 0.8}     # 정상/관찰 10%
        }
        
        # 상황 분석 표현
        self.situation_expressions = {
            "detection_verbs": [
                "감지되었습니다", "탐지되었습니다", "확인되었습니다",
                "포착되었습니다", "관측되었습니다", "발견되었습니다"
            ],
            "over_standard": [
                "기준 {baseline}{unit}를 {diff}{unit} 초과하여 이상 행동으로 판단됩니다",
                "기준 {baseline}{unit}를 {diff}{unit} 넘어서 비정상 이동으로 평가됩니다",
                "기준치 {baseline}{unit}를 {diff}{unit} 상회하여 위험 상황입니다"
            ],
            "within_standard": [
                "측정값이 기준 {baseline}{unit}에 부합하여 정상 범위로 평가됩니다",
                "기준 {baseline}{unit} 범위에서 정상 이동으로 판단됩니다",
                "기준치 {baseline}{unit} 내에서 양호한 상태를 유지하고 있습니다"
            ],
            "no_standard": [
                "기준이 정의되지 않아 추가 판단이 불가하니 관찰 상태를 유지하세요",
                "허용 기준이 미설정되어 즉각적 조치 대신 관찰만 실시하십시오",
                "기준치가 설정되지 않은 구역이므로 모니터링만 진행하십시오",
                "기준치가 설정되지 않아 관찰만 유지하세요"
            ]
        }
        
        # 조치 표현
        self.action_expressions = {
            "immediate": [
                "보안팀은 즉시 현장 확인하세요",
                "보안요원의 즉시 점검이 필요합니다",
                "긴급 대응이 필요합니다",
                "즉각적인 조치가 요구됩니다"
            ],
            "warning": [
                "위험 경보가 필요합니다",
                "주의가 요구됩니다",
                "경계 강화가 필요합니다"
            ],
            "monitoring": [
                "반복 시 점검을 권장합니다",
                "지속 모니터링을 권장합니다",
                "향후 관찰이 필요합니다",
                "계속적인 모니터링이 요구됩니다"
            ],
            "observation": [
                "지속 모니터링을 권장합니다",
                "관찰만 유지하세요",
                "모니터링만 진행하십시오"
            ]
        }
        
        # 추가 설명 표현 (3-4문장 구성용)
        self.additional_explanations = [
            "해당 행동은 비정상 이동 패턴으로 평가됩니다",
            "측정된 이동 패턴이 기준치를 초과한 원인 분석이 필요합니다",
            "{pattern} 패턴이 지속적으로 발생한 것으로 확인됩니다"
        ]
        
        # Input 구조형/설명형 패턴
        self.input_patterns = {
            "structured": {  # 구조형 60%
                "weight": 60,
                "templates": [
                    "{time} {location}에서 {object}이 {pattern} {measurement}, {baseline}",
                    "{baseline}, {time} {location}에서 {object}이 {pattern} {measurement}",
                    "{object}이 {time}에 {location}에서 {pattern} {measurement}, {baseline}",
                    "{location}에서 {time}에 {object}이 {pattern} {measurement}, {baseline}",
                    "{time}에 {object}이 {pattern} 후 {measurement} 이동했습니다, {baseline}"
                ]
            },
            "descriptive": {  # 설명형 40%
                "weight": 40,
                "templates": [
                    "{time}에 {object}이 {pattern} 행동을 보였으며 {measurement} 머물렀습니다, {baseline}",
                    "{object}이 {time}에 {location}에서 {pattern} 동작을 보였으며 {measurement} 머물렀습니다, {baseline}",
                    "{baseline}, {time} {object}이 {location}에서 {pattern}을 {measurement} 수행했습니다",
                    "{time}에 {object}이 {pattern} 후 {measurement} 이동했습니다, {baseline}"
                ]
            }
        }
        
    def generate_time_format(self) -> str:
        """시간 형식 생성 (HH:MM 50% / 한글 시간 50%)"""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        if random.random() < 0.5:
            # HH:MM 형식 (50%)
            return f"{hour:02d}:{minute:02d}"
        else:
            # 한글 시간 형식 (50%)
            time_labels = {
                0: "자정", 1: "새벽", 2: "아침", 3: "아침", 4: "아침", 5: "아침",
                6: "오전", 7: "오전", 8: "오전", 9: "오전", 10: "오전", 11: "오전",
                12: "정오", 13: "오후", 14: "오후", 15: "오후", 16: "오후", 17: "오후",
                18: "저녁", 19: "저녁", 20: "저녁", 21: "저녁", 22: "밤", 23: "밤"
            }
            
            label = time_labels[hour]
            if hour == 0:
                return f"{label} 12시 {minute}분"
            elif hour == 12:
                return f"{label} 12시 {minute}분"
            elif hour > 12:
                return f"{label} {hour-12}시 {minute}분"
            else:
                return f"{label} {hour}시 {minute}분"
    
    def generate_location(self) -> str:
        """장소 생성 (단순 장소 30-40% / 장소+구역명 60-70%)"""
        if random.random() < 0.35:  # 35% 단순 장소
            return random.choice(self.locations["simple"])
        else:  # 65% 장소+구역명
            return random.choice(self.locations["complex"])
    
    def generate_object(self) -> str:
        """객체 생성 (사람 또는 차량)"""
        if random.random() < 0.7:  # 70% 사람
            person = random.choice(self.person_types)
            return f"{person} 1명"
        else:  # 30% 차량
            vehicle = random.choice(self.vehicle_types)
            return f"{vehicle} 1대"
    
    def generate_pattern_and_measurement(self) -> Dict:
        """패턴과 측정값 생성"""
        # 패턴 선택
        pattern_weights = [self.movement_patterns[p]["weight"] for p in self.movement_patterns.keys()]
        pattern = random.choices(list(self.movement_patterns.keys()), weights=pattern_weights)[0]
        
        # 측정 단위 선택 (거리 또는 시간)
        unit_type = random.choice(list(self.measurement_units.keys()))
        unit_info = self.measurement_units[unit_type]
        
        # 측정값 범위 선택
        range_weights = [unit_info["ranges"][r]["weight"] for r in unit_info["ranges"].keys()]
        selected_range = random.choices(list(unit_info["ranges"].keys()), weights=range_weights)[0]
        range_info = unit_info["ranges"][selected_range]
        
        # 측정값 생성
        value = random.randint(range_info["min"], range_info["max"])
        
        # 측정값 형식 선택 (N초 70% vs N초간 30%)
        format_choice = random.choices(unit_info["formats"], weights=[70, 30])[0]
        measurement_text = format_choice.format(value=value)
        
        return {
            "pattern": pattern,
            "unit_type": unit_type,
            "unit": unit_info["unit"],
            "value": value,
            "measurement_text": measurement_text
        }
    
    def generate_baseline(self, measurement_info: Dict) -> Dict:
        """기준값 생성"""
        # 기준 설정 여부 결정
        baseline_weights = [self.baseline_types[bt]["weight"] for bt in self.baseline_types.keys()]
        baseline_type = random.choices(list(self.baseline_types.keys()), weights=baseline_weights)[0]
        
        if baseline_type == "기준있음":
            # 기준값 생성
            current_value = measurement_info["value"]
            
            # 위험도 분류에 따른 기준값 설정
            risk_weights = [self.risk_categories[r]["weight"] for r in self.risk_categories.keys()]
            risk_type = random.choices(list(self.risk_categories.keys()), weights=risk_weights)[0]
            
            if risk_type == "위험":  # 기준 초과 (70%)
                baseline_value = random.randint(max(1, current_value - 8), current_value - 1)
            elif risk_type == "정상":  # 기준 내 (10%)
                baseline_value = random.randint(current_value + 1, current_value + 8)
            else:  # 주의 (20%) - 거의 기준에 근접
                if random.random() < 0.5:
                    baseline_value = current_value  # 정확히 기준과 동일
                else:
                    baseline_value = random.randint(max(1, current_value - 2), current_value + 2)
            
            format_template = random.choice(self.baseline_types[baseline_type]["formats"])
            baseline_text = format_template.format(value=baseline_value, unit=measurement_info["unit"])
            
            return {
                "has_baseline": True,
                "baseline_value": baseline_value,
                "baseline_text": baseline_text,
                "is_over": current_value > baseline_value,
                "risk_type": risk_type
            }
        else:
            # 기준 없음
            baseline_text = random.choice(self.baseline_types[baseline_type]["formats"])
            return {
                "has_baseline": False,
                "baseline_value": None,
                "baseline_text": baseline_text,
                "is_over": False,
                "risk_type": "기준없음"
            }
    
    def generate_input_text(self, input_data: Dict) -> str:
        """Input 텍스트 생성"""
        # 구조형/설명형 선택
        input_weights = [self.input_patterns[t]["weight"] for t in self.input_patterns.keys()]
        input_type = random.choices(list(self.input_patterns.keys()), weights=input_weights)[0]
        
        # 템플릿 선택
        template = random.choice(self.input_patterns[input_type]["templates"])
        
        # 템플릿에 데이터 적용
        input_text = template.format(
            time=input_data["time"],
            location=input_data["location"],
            object=input_data["object"],
            pattern=input_data["pattern_info"]["pattern"],
            measurement=input_data["pattern_info"]["measurement_text"],
            baseline=input_data["baseline_info"]["baseline_text"]
        )
        
        return input_text
    
    def generate_situation_analysis(self, input_data: Dict) -> str:
        """상황 분석 문장 생성"""
        time = input_data["time"]
        location = input_data["location"]
        object_str = input_data["object"]
        pattern_info = input_data["pattern_info"]
        
        # 감지 동사 선택
        detection_verb = random.choice(self.situation_expressions["detection_verbs"])
        
        # 기본 상황 문장 (시작점 다양화)
        start_patterns = [
            f"{time}에 {location}에서 {object_str}이 {pattern_info['measurement_text']} {pattern_info['pattern']} {detection_verb}",  # 40%
            f"{location}에서 {time}에 {object_str}이 {pattern_info['measurement_text']} {pattern_info['pattern']} {detection_verb}",   # 20%
            f"{object_str}이 {time}에 {location}에서 {pattern_info['measurement_text']} {pattern_info['pattern']} {detection_verb}",   # 20%
            f"{pattern_info['measurement_text']} {pattern_info['pattern']}이 {time}에 {location}에서 {detection_verb}"  # 20%
        ]
        
        weights = [40, 20, 20, 20]
        situation_base = random.choices(start_patterns, weights=weights)[0]
        
        return situation_base
    
    def generate_evaluation_sentence(self, input_data: Dict) -> str:
        """평가 문장 생성"""
        pattern_info = input_data["pattern_info"]
        baseline_info = input_data["baseline_info"]
        
        if not baseline_info["has_baseline"]:
            # 기준 없음
            return random.choice(self.situation_expressions["no_standard"])
        elif baseline_info["is_over"]:
            # 기준 초과
            diff = pattern_info["value"] - baseline_info["baseline_value"]
            expr = random.choice(self.situation_expressions["over_standard"])
            return expr.format(
                baseline=baseline_info["baseline_value"],
                unit=pattern_info["unit"],
                diff=diff
            )
        else:
            # 기준 내
            expr = random.choice(self.situation_expressions["within_standard"])
            return expr.format(
                baseline=baseline_info["baseline_value"],
                unit=pattern_info["unit"]
            )
    
    def generate_action_sentence(self, input_data: Dict) -> str:
        """조치 문장 생성"""
        baseline_info = input_data["baseline_info"]
        
        if not baseline_info["has_baseline"]:
            # 기준 없음 - 관찰
            return random.choice(self.action_expressions["observation"])
        elif baseline_info["is_over"]:
            # 기준 초과 - 즉시 조치 또는 경고
            if random.random() < 0.8:  # 80% 즉시 조치
                return random.choice(self.action_expressions["immediate"])
            else:  # 20% 경고
                return random.choice(self.action_expressions["warning"])
        else:
            # 기준 내 - 모니터링
            return random.choice(self.action_expressions["monitoring"])
    
    def generate_natural_output(self, input_data: Dict) -> str:
        """자연스러운 2~4문장 Output 생성"""
        situation = self.generate_situation_analysis(input_data)
        evaluation = self.generate_evaluation_sentence(input_data)
        action = self.generate_action_sentence(input_data)
        
        # 문장 수 결정 (2문장: 40%, 3문장: 45%, 4문장: 15%)
        sentence_count = random.choices([2, 3, 4], weights=[40, 45, 15])[0]
        
        if sentence_count == 2:
            # 2문장: [상황] + [평가+조치]
            return f"{situation}. {evaluation}. {action}."
        elif sentence_count == 3:
            # 3문장 패턴 선택
            patterns = [
                # 패턴 1: [상황] + [평가] + [조치]
                lambda: f"{situation}. {evaluation}. {action}.",
                
                # 패턴 2: [상황] + [추가설명] + [평가+조치]
                lambda: f"{situation}. {random.choice(self.additional_explanations).format(pattern=input_data['pattern_info']['pattern'])}. {evaluation}. {action}.",
                
                # 패턴 3: [평가] + [조치] + [상황]
                lambda: f"{evaluation}. {action}. {situation}."
            ]
            
            selected_pattern = random.choice(patterns)
            return selected_pattern()
        else:  # 4문장
            # 4문장: [상황] + [추가설명] + [평가] + [조치]
            additional = random.choice(self.additional_explanations).format(pattern=input_data['pattern_info']['pattern'])
            return f"{situation}. {additional}. {evaluation}. {action}."
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        # 기본 데이터 생성
        time = self.generate_time_format()
        location = self.generate_location()
        object_str = self.generate_object()
        pattern_info = self.generate_pattern_and_measurement()
        baseline_info = self.generate_baseline(pattern_info)
        
        input_data = {
            "time": time,
            "location": location,
            "object": object_str,
            "pattern_info": pattern_info,
            "baseline_info": baseline_info
        }
        
        # Input 텍스트 생성
        input_text = self.generate_input_text(input_data)
        
        # Output 텍스트 생성
        output_text = self.generate_natural_output(input_data)
        
        # 도메인명
        domain = "이상 이동 패턴 감지"
        
        return input_text, output_text, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인10 데이터셋 {count}개 생성 중...")
        
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
        # 패턴 유형 분포
        pattern_distribution = {p: 0 for p in self.movement_patterns.keys()}
        
        # 객체 유형 분포
        object_distribution = {"사람": 0, "차량": 0}
        
        # 측정 단위 분포
        unit_distribution = {"거리(m)": 0, "시간(초)": 0}
        
        # 기준 설정 분포
        baseline_distribution = {"기준있음": 0, "기준없음": 0}
        
        # 시간 형식 분포
        time_formats = {"HH:MM": 0, "한글시간": 0}
        
        # 장소 형식 분포
        location_types = {"단순장소": 0, "장소+구역명": 0}
        
        # Input 형식 분포
        input_types = {"구조형": 0, "설명형": 0}
        
        # 문장 길이 분포
        sentence_lengths = {"2문장": 0, "3문장": 0, "4문장": 0, "기타": 0}
        
        # 위험도 분포
        risk_distribution = {"위험상황": 0, "정상상황": 0, "기준없음상황": 0}
        
        for input_str, output_str, domain in dataset:
            # 패턴 유형 체크
            for pattern in self.movement_patterns.keys():
                if pattern in input_str:
                    pattern_distribution[pattern] += 1
                    break
            
            # 객체 유형 체크
            if any(person in input_str for person in self.person_types):
                object_distribution["사람"] += 1
            else:
                object_distribution["차량"] += 1
            
            # 측정 단위 체크
            if "m" in input_str:
                unit_distribution["거리(m)"] += 1
            elif "초" in input_str:
                unit_distribution["시간(초)"] += 1
            
            # 기준 설정 체크
            if any(phrase in input_str for phrase in ["기준:", "기준 "]):
                baseline_distribution["기준있음"] += 1
            else:
                baseline_distribution["기준없음"] += 1
            
            # 시간 형식 체크
            if re.search(r'\d{2}:\d{2}', input_str):
                time_formats["HH:MM"] += 1
            else:
                time_formats["한글시간"] += 1
            
            # 장소 형식 체크
            has_complex_location = False
            for complex_loc in self.locations["complex"]:
                if complex_loc in input_str:
                    has_complex_location = True
                    break
            
            if has_complex_location:
                location_types["장소+구역명"] += 1
            else:
                location_types["단순장소"] += 1
            
            # Input 형식 체크 (설명형 키워드 확인)
            if any(keyword in input_str for keyword in ["보였으며", "머물렀습니다", "동작을", "수행했습니다"]):
                input_types["설명형"] += 1
            else:
                input_types["구조형"] += 1
            
            # 문장 길이 체크 (마침표 기준)
            sentence_count = len([s for s in output_str.split('.') if s.strip()])
            
            if sentence_count == 2:
                sentence_lengths["2문장"] += 1
            elif sentence_count == 3:
                sentence_lengths["3문장"] += 1
            elif sentence_count == 4:
                sentence_lengths["4문장"] += 1
            else:
                sentence_lengths["기타"] += 1
            
            # 위험도 체크
            if any(word in output_str for word in ["초과", "넘어", "상회", "즉시", "긴급"]):
                risk_distribution["위험상황"] += 1
            elif any(word in output_str for word in ["부합", "정상", "양호"]):
                risk_distribution["정상상황"] += 1
            else:
                risk_distribution["기준없음상황"] += 1
        
        total_count = len(dataset)
        
        return {
            "total_count": total_count,
            "pattern_distribution": {
                p: f"{pattern_distribution[p]} ({pattern_distribution[p]/total_count*100:.1f}%)"
                for p in self.movement_patterns.keys()
            },
            "object_distribution": {
                "사람": f"{object_distribution['사람']} ({object_distribution['사람']/total_count*100:.1f}%)",
                "차량": f"{object_distribution['차량']} ({object_distribution['차량']/total_count*100:.1f}%)"
            },
            "unit_distribution": {
                "거리(m)": f"{unit_distribution['거리(m)']} ({unit_distribution['거리(m)']/total_count*100:.1f}%)",
                "시간(초)": f"{unit_distribution['시간(초)']} ({unit_distribution['시간(초)']/total_count*100:.1f}%)"
            },
            "baseline_distribution": {
                "기준있음": f"{baseline_distribution['기준있음']} ({baseline_distribution['기준있음']/total_count*100:.1f}%)",
                "기준없음": f"{baseline_distribution['기준없음']} ({baseline_distribution['기준없음']/total_count*100:.1f}%)"
            },
            "time_formats": {
                "HH:MM": f"{time_formats['HH:MM']} ({time_formats['HH:MM']/total_count*100:.1f}%)",
                "한글시간": f"{time_formats['한글시간']} ({time_formats['한글시간']/total_count*100:.1f}%)"
            },
            "location_types": {
                "단순장소": f"{location_types['단순장소']} ({location_types['단순장소']/total_count*100:.1f}%)",
                "장소+구역명": f"{location_types['장소+구역명']} ({location_types['장소+구역명']/total_count*100:.1f}%)"
            },
            "input_types": {
                "구조형": f"{input_types['구조형']} ({input_types['구조형']/total_count*100:.1f}%)",
                "설명형": f"{input_types['설명형']} ({input_types['설명형']/total_count*100:.1f}%)"
            },
            "sentence_lengths": {
                "2문장": f"{sentence_lengths['2문장']} ({sentence_lengths['2문장']/total_count*100:.1f}%)",
                "3문장": f"{sentence_lengths['3문장']} ({sentence_lengths['3문장']/total_count*100:.1f}%)",
                "4문장": f"{sentence_lengths['4문장']} ({sentence_lengths['4문장']/total_count*100:.1f}%)",
                "기타": f"{sentence_lengths['기타']} ({sentence_lengths['기타']/total_count*100:.1f}%)"
            },
            "risk_distribution": {
                "위험상황": f"{risk_distribution['위험상황']} ({risk_distribution['위험상황']/total_count*100:.1f}%)",
                "정상상황": f"{risk_distribution['정상상황']} ({risk_distribution['정상상황']/total_count*100:.1f}%)",
                "기준없음상황": f"{risk_distribution['기준없음상황']} ({risk_distribution['기준없음상황']/total_count*100:.1f}%)"
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
    generator = Domain10AbnormalMovementGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 저장 경로 설정
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain10_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 데이터셋 검증 및 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인10 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    
    print(f"\n🎭 이상 이동 패턴 분포:")
    for pattern, stats in validation_results['pattern_distribution'].items():
        print(f"   {pattern}: {stats}")
    
    print(f"\n👥 객체 유형 분포:")
    print(f"   사람: {validation_results['object_distribution']['사람']}")
    print(f"   차량: {validation_results['object_distribution']['차량']}")
    
    print(f"\n📏 측정 단위 분포:")
    print(f"   거리(m): {validation_results['unit_distribution']['거리(m)']}")
    print(f"   시간(초): {validation_results['unit_distribution']['시간(초)']}")
    
    print(f"\n📋 기준 설정 분포:")
    print(f"   기준있음: {validation_results['baseline_distribution']['기준있음']}")
    print(f"   기준없음: {validation_results['baseline_distribution']['기준없음']}")
    
    print(f"\n⏰ 시간 형식 분포:")
    print(f"   HH:MM 형식: {validation_results['time_formats']['HH:MM']}")
    print(f"   한글 시간: {validation_results['time_formats']['한글시간']}")
    
    print(f"\n🏢 장소 형식 분포:")
    print(f"   단순장소: {validation_results['location_types']['단순장소']}")
    print(f"   장소+구역명: {validation_results['location_types']['장소+구역명']}")
    
    print(f"\n📝 Input 형식 분포:")
    print(f"   구조형: {validation_results['input_types']['구조형']}")
    print(f"   설명형: {validation_results['input_types']['설명형']}")
    
    print(f"\n📄 문장 길이 분포:")
    print(f"   2문장: {validation_results['sentence_lengths']['2문장']}")
    print(f"   3문장: {validation_results['sentence_lengths']['3문장']}")
    print(f"   4문장: {validation_results['sentence_lengths']['4문장']}")
    print(f"   기타: {validation_results['sentence_lengths']['기타']}")
    
    print(f"\n⚠️ 위험도 분포:")
    print(f"   위험상황: {validation_results['risk_distribution']['위험상황']}")
    print(f"   정상상황: {validation_results['risk_distribution']['정상상황']}")
    print(f"   기준없음상황: {validation_results['risk_distribution']['기준없음상황']}") 