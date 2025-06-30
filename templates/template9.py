#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
도메인9 줄 서기 및 대기열 정렬 상태 감지 데이터셋 생성기
template8.py 구조를 기반으로 한 Domain9 전용 생성기

주요 기능:
- 7가지 행동 유형별 정렬 상태 감지 데이터 생성
- 거리/시간 기반 측정값과 기준값 비교
- 다양한 장소와 대상별 상황 시뮬레이션
- 기준 설정/미설정 상황 처리
- 2~3문장 자연스러운 Output 생성
- 1000개 대용량 데이터셋 생성
"""

import random
import csv
import json
import os
import re
from typing import List, Dict, Tuple, Set
from collections import Counter

class Domain9QueueAlignmentGenerator:
    """
    도메인9 줄 서기 및 대기열 정렬 상태 감지 데이터셋 생성기
    """
    
    def __init__(self):
        """생성기 초기화 - 모든 설정값과 패턴 정의"""
        
        # 행동 유형별 설정 (7가지 주요 행동)
        self.behavior_types = {
            "흐트러짐": {"weight": 20, "unit_types": ["distance", "time"]},
            "밀집": {"weight": 18, "unit_types": ["distance", "time"]},
            "병렬정렬": {"weight": 16, "unit_types": ["distance", "time"]},
            "역방향섞임": {"weight": 14, "unit_types": ["distance", "time"]},
            "군집": {"weight": 12, "unit_types": ["distance", "time"]},
            "일정한간격유지": {"weight": 10, "unit_types": ["distance", "time"]},
            "일정방향정렬": {"weight": 10, "unit_types": ["distance", "time"]}
        }
        
        # 측정 단위 설정
        self.measurement_units = {
            "distance": {
                "unit": "m",
                "ranges": {
                    "짧음": {"min": 0.5, "max": 1.5, "weight": 40},
                    "보통": {"min": 1.6, "max": 2.5, "weight": 35},
                    "김": {"min": 2.6, "max": 3.0, "weight": 25}
                }
            },
            "time": {
                "unit": "초",
                "ranges": {
                    "짧음": {"min": 2, "max": 5, "weight": 35},
                    "보통": {"min": 6, "max": 8, "weight": 40},
                    "김": {"min": 9, "max": 10, "weight": 25}
                }
            }
        }
        
        # 장소별 설정 (대기줄이 형성되는 다양한 장소들)
        self.locations = [
            # 교통 관련 시설
            "공항 대기구역", "지하철역", "버스 정류장 앞", "기차역", "택시 승강장", 
            "고속버스터미널", "지하철 개찰구", "공항 체크인 카운터", "공항 보안검색대", 
            "항공기 탑승게이트", "기차 승강장", "버스 터미널 대기실", "페리 터미널",
            
            # 의료 관련 시설
            "병원 접수창구", "의료센터", "종합병원", "응급실 대기실", "약국 앞", 
            "검사실 앞", "치과 대기실", "한의원", "보건소", "건강검진센터", 
            "재활센터", "산부인과", "소아과 대기실", "안과 접수처",
            
            # 상업 관련 시설
            "매점 앞", "카페 앞", "백화점", "쇼핑몰", "편의점", "마트 계산대", 
            "대형마트 입구", "패스트푸드점", "레스토랑 입구", "푸드코트", "서점", 
            "은행 창구", "ATM 앞", "우체국", "택배 집하장", "드라이브스루",
            
            # 문화/교육/엔터테인먼트 시설
            "공연장 입구", "박물관", "도서관", "영화관 매표소", "콘서트홀", "미술관",
            "체육관 입구", "수영장", "놀이공원 입구", "동물원", "아쿠아리움", 
            "과학관", "전시관", "컨벤션센터", "스포츠센터", "볼링장", "노래방",
            
            # 교육 기관
            "학교 급식실", "대학교 학생식당", "도서관 열람실", "강의실 앞", 
            "시험장 입구", "학원 접수처", "어학원", "컴퓨터학원", "음악학원",
            
            # 공공기관/업무 시설
            "사무실", "로비", "입구", "출구", "엘리베이터 앞", "회의실 앞", 
            "면접 대기실", "구청", "시청", "동사무소", "경찰서", "법원", 
            "세무서", "고용센터", "사회보험공단", "국민연금공단",
            
            # 금융 관련 시설
            "은행 본점", "신용협동조합", "증권회사", "보험회사", "대출상담센터",
            "외환은행", "투자상담소", "부동산중개소",
            
            # 숙박/관광 관련 시설
            "호텔 체크인", "모텔 프런트", "펜션 접수처", "게스트하우스", 
            "리조트 로비", "관광안내소", "여행사", "렌터카 사무소", "면세점",
            
            # 서비스업 관련 시설
            "미용실", "네일샵", "마사지샵", "사우나", "찜질방", "세탁소", 
            "휴대폰 대리점", "통신사 매장", "AS센터", "서비스센터",
            
            # 종교 시설
            "교회", "성당", "절", "성전", "기도원", "종교센터",
            
            # 레저/스포츠 시설
            "골프장 클럽하우스", "테니스장", "스키장 리프트", "캠핑장 접수처",
            "해수욕장 안내소", "공원 입구", "산책로", "자전거 대여소",
            
            # 기타 구역
            "A구역", "B구역", "C구역", "D구역", "E구역", "F구역", 
            "1층 로비", "2층 대기실", "지하 1층", "옥상 정원", "중앙광장"
        ]
        
        # 대상 인물 유형
        self.person_types = ["환자", "학생", "고객", "승객", "방문자"]
        
        # 기준 설정 유형 (50% 기준있음, 50% 기준없음)
        self.baseline_types = {
            "기준있음": {"weight": 50, "formats": ["기준: {value}{unit}", "기준 {value}{unit}"]},
            "기준없음": {"weight": 50, "formats": ["허용치 없음", "기준 미설정", "기준 없음", "허용치 미설정"]}
        }
        
        # 상황 분석 표현
        self.situation_expressions = {
            "over_standard": [
                "기준 {baseline}{unit}를 {diff}{unit} 초과하여 정렬 이상으로 판단됩니다",
                "기준 {baseline}{unit}를 {diff}{unit} 넘어서 비정상 정렬로 평가됩니다",
                "기준치 {baseline}{unit}를 {diff}{unit} 상회하여 정렬 문제가 감지됩니다"
            ],
            "within_standard": [
                "측정값이 기준 {baseline}{unit}에 부합하여 정상 정렬로 평가됩니다",
                "기준 {baseline}{unit} 범위에서 정상 정렬 상태입니다",
                "기준치 {baseline}{unit} 내에서 양호한 정렬을 유지하고 있습니다"
            ],
            "no_standard": [
                "기준치가 설정되지 않아 관찰만 유지하세요",
                "허용 기준이 미설정되어 즉각적 조치 대신 관찰만 실시하십시오",
                "기준이 정의되지 않아 추가 판단이 불가하니 관찰 상태를 유지하세요",
                "기준치가 설정되지 않은 구역이므로 모니터링만 진행하십시오"
            ]
        }
        
        # 조치 표현
        self.action_expressions = {
            "immediate": [
                "보안요원의 즉시 점검이 필요합니다",
                "보안요원은 즉시 점검하세요",
                "즉각적인 정렬 조치가 필요합니다"
            ],
            "monitoring": [
                "향후 반복 발생 시 점검이 권장됩니다",
                "반복 시 점검을 권장합니다",
                "지속적인 모니터링이 권장됩니다",
                "지속 모니터링을 권장됩니다"
            ],
            "observation": [
                "관찰을 유지해주세요",
                "관찰만 실시하십시오",
                "모니터링만 진행하십시오"
            ]
        }
        
        # 감지 동사 다양화 (기존 "감지되었습니다" 대신)
        self.detection_verbs = [
            "확인되었습니다",
            "포착되었습니다", 
            "관측되었습니다",
            "발견되었습니다",
            "감지되었습니다"
        ]
        
        # 추가 설명 표현 (3문장 구성용)
        self.additional_explanations = [
            "행동 패턴이 지속적으로 발생한 것으로 확인됩니다",
            "측정된 정렬 상태가 기준치를 초과한 원인 분석이 필요합니다",
            "해당 행동은 비정상 정렬 상태로 평가됩니다"
        ]
        
    def generate_time_format(self) -> str:
        """시간 형식 생성 (50:50 비율)"""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        if random.random() < 0.5:
            # HH:MM 형식
            return f"{hour:02d}:{minute:02d}"
        else:
            # 한글 시간 형식
            if hour < 12:
                if hour == 0:
                    return f"자정 12시 {minute}분"
                else:
                    return f"오전 {hour}시 {minute}분"
            elif hour == 12:
                return f"정오 12시 {minute}분"
            else:
                time_labels = ["오후", "저녁", "밤"]
                label = random.choice(time_labels)
                display_hour = hour - 12 if hour > 12 else hour
                return f"{label} {display_hour}시 {minute}분"
    
    def generate_behavior_and_measurement(self) -> Dict:
        """행동 유형과 측정값 생성"""
        # 행동 유형 선택
        behavior_weights = [self.behavior_types[bt]["weight"] for bt in self.behavior_types.keys()]
        behavior = random.choices(list(self.behavior_types.keys()), weights=behavior_weights)[0]
        
        # 측정 단위 선택 (거리 또는 시간)
        unit_type = random.choice(list(self.measurement_units.keys()))
        unit_info = self.measurement_units[unit_type]
        
        # 측정값 범위 선택
        range_weights = [unit_info["ranges"][r]["weight"] for r in unit_info["ranges"].keys()]
        selected_range = random.choices(list(unit_info["ranges"].keys()), weights=range_weights)[0]
        range_info = unit_info["ranges"][selected_range]
        
        # 측정값 생성
        if unit_type == "distance":
            value = round(random.uniform(range_info["min"], range_info["max"]), 1)
        else:  # time
            value = random.randint(range_info["min"], range_info["max"])
        
        return {
            "behavior": behavior,
            "unit_type": unit_type,
            "unit": unit_info["unit"],
            "value": value
        }
    
    def generate_baseline(self, measurement_info: Dict) -> Dict:
        """기준값 생성"""
        # 기준 설정 여부 결정
        baseline_weights = [self.baseline_types[bt]["weight"] for bt in self.baseline_types.keys()]
        baseline_type = random.choices(list(self.baseline_types.keys()), weights=baseline_weights)[0]
        
        if baseline_type == "기준있음":
            # 기준값 생성 (측정값보다 작거나 큰 값)
            current_value = measurement_info["value"]
            unit_type = measurement_info["unit_type"]
            
            if unit_type == "distance":
                if random.random() < 0.6:  # 60% 확률로 기준값이 더 작음 (초과 상황)
                    baseline_value = round(random.uniform(0.5, max(0.6, current_value - 0.1)), 1)
                else:  # 40% 확률로 기준값이 더 큼 (정상 상황)
                    baseline_value = round(random.uniform(current_value + 0.1, max(current_value + 0.2, 3.0)), 1)
            else:  # time
                if random.random() < 0.6:  # 60% 확률로 기준값이 더 작음 (초과 상황)
                    baseline_value = random.randint(2, max(2, current_value - 1))
                else:  # 40% 확률로 기준값이 더 큼 (정상 상황)
                    baseline_value = random.randint(current_value + 1, max(current_value + 2, 10))
            
            format_template = random.choice(self.baseline_types[baseline_type]["formats"])
            baseline_text = format_template.format(value=baseline_value, unit=measurement_info["unit"])
            
            return {
                "has_baseline": True,
                "baseline_value": baseline_value,
                "baseline_text": baseline_text,
                "is_over": current_value > baseline_value
            }
        else:
            # 기준 없음
            baseline_text = random.choice(self.baseline_types[baseline_type]["formats"])
            return {
                "has_baseline": False,
                "baseline_value": None,
                "baseline_text": baseline_text,
                "is_over": False
            }
    
    def generate_input_format(self) -> Dict:
        """Input 형식 생성"""
        time = self.generate_time_format()
        location = random.choice(self.locations)
        person_type = random.choice(self.person_types)
        measurement_info = self.generate_behavior_and_measurement()
        baseline_info = self.generate_baseline(measurement_info)
        
        # Input 문자열 패턴 선택
        patterns = [
            # 패턴 1: 시간에 인물이 행동하며 측정값 유지했습니다, 기준
            f"{time}에 {person_type} 1명이 {measurement_info['behavior']}하며 {measurement_info['value']}{measurement_info['unit']} 유지했습니다, {baseline_info['baseline_text']}",
            
            # 패턴 2: 행동이 시간에 장소에서 감지되어 측정값 유지되었습니다, 기준
            f"{measurement_info['behavior']}이 {time}에 {location}에서 감지되어 {measurement_info['value']}{measurement_info['unit']} 유지되었습니다, {baseline_info['baseline_text']}",
            
            # 패턴 3: 인물이 시간에 장소에서 행동 측정값, 기준
            f"{person_type} 1명이 {time}에 {location}에서 {measurement_info['behavior']} {measurement_info['value']}{measurement_info['unit']}, {baseline_info['baseline_text']}",
            
            # 패턴 4: 기준: 값, 시간에 장소에서 인물이 행동 측정값
            f"{baseline_info['baseline_text']}, {time}에 {person_type} 1명이 {measurement_info['behavior']} 후 {measurement_info['value']}{measurement_info['unit']} 유지했습니다",
            
            # 패턴 5: 장소에서 시간에 인물이 행동 측정값, 기준
            f"{location}에서 {time}에 {person_type} 1명이 {measurement_info['behavior']} {measurement_info['value']}{measurement_info['unit']}, {baseline_info['baseline_text']}"
        ]
        
        input_text = random.choice(patterns)
        
        return {
            "input_text": input_text,
            "time": time,
            "location": location,
            "person_type": person_type,
            "measurement_info": measurement_info,
            "baseline_info": baseline_info
        }
    
    def generate_situation_analysis(self, input_data: Dict) -> str:
        """상황 분석 문장 생성"""
        time = input_data["time"]
        location = input_data["location"]
        person_type = input_data["person_type"]
        measurement_info = input_data["measurement_info"]
        baseline_info = input_data["baseline_info"]
        
        # 감지 동사 선택
        detection_verb = random.choice(self.detection_verbs)
        
        # 기본 상황 문장
        if measurement_info["unit_type"] == "time":
            situation_base = f"{time}에 {location}에서 {person_type} 1명이 {measurement_info['value']}{measurement_info['unit']} {measurement_info['behavior']} {detection_verb}"
        else:
            situation_base = f"{time}에 {location}에서 {person_type} 1명이 {measurement_info['value']}{measurement_info['unit']} {measurement_info['behavior']} {detection_verb}"
        
        return situation_base
    
    def generate_evaluation_sentence(self, input_data: Dict) -> str:
        """평가 문장 생성"""
        measurement_info = input_data["measurement_info"]
        baseline_info = input_data["baseline_info"]
        
        if not baseline_info["has_baseline"]:
            # 기준 없음
            return random.choice(self.situation_expressions["no_standard"])
        elif baseline_info["is_over"]:
            # 기준 초과
            diff = measurement_info["value"] - baseline_info["baseline_value"]
            if measurement_info["unit_type"] == "distance":
                diff = round(diff, 1)
            else:
                diff = int(diff)
            
            expr = random.choice(self.situation_expressions["over_standard"])
            return expr.format(
                baseline=baseline_info["baseline_value"],
                unit=measurement_info["unit"],
                diff=diff
            )
        else:
            # 기준 내
            expr = random.choice(self.situation_expressions["within_standard"])
            return expr.format(
                baseline=baseline_info["baseline_value"],
                unit=measurement_info["unit"]
            )
    
    def generate_action_sentence(self, input_data: Dict) -> str:
        """조치 문장 생성"""
        baseline_info = input_data["baseline_info"]
        
        if not baseline_info["has_baseline"]:
            # 기준 없음 - 관찰
            return random.choice(self.action_expressions["observation"])
        elif baseline_info["is_over"]:
            # 기준 초과 - 즉시 조치
            return random.choice(self.action_expressions["immediate"])
        else:
            # 기준 내 - 모니터링
            return random.choice(self.action_expressions["monitoring"])
    
    def generate_natural_output(self, input_data: Dict) -> str:
        """자연스러운 2~3문장 Output 생성"""
        situation = self.generate_situation_analysis(input_data)
        evaluation = self.generate_evaluation_sentence(input_data)
        action = self.generate_action_sentence(input_data)
        
        # 문장 수 결정 (2문장: 60%, 3문장: 40%)
        sentence_count = random.choices([2, 3], weights=[60, 40])[0]
        
        if sentence_count == 2:
            # 2문장: [상황] + [평가+조치]
            return f"{situation}. {evaluation}. {action}."
        else:
            # 3문장 패턴 선택
            patterns = [
                # 패턴 1: [상황] + [평가] + [조치]
                lambda: f"{situation}. {evaluation}. {action}.",
                
                # 패턴 2: [상황] + [추가설명] + [평가+조치]
                lambda: f"{situation}. {random.choice(self.additional_explanations)}. {evaluation}. {action}.",
                
                # 패턴 3: [상황] + [평가] + [추가설명+조치]
                lambda: f"{situation}. {evaluation}. {random.choice(self.additional_explanations)}. {action}."
            ]
            
            selected_pattern = random.choice(patterns)
            return selected_pattern()
    
    def generate_single_data(self) -> Tuple[str, str, str]:
        """단일 데이터 생성"""
        input_data = self.generate_input_format()
        output_text = self.generate_natural_output(input_data)
        domain = "줄 서기 및 대기열 정렬 상태 감지"
        
        return input_data["input_text"], output_text, domain
    
    def generate_dataset(self, count: int) -> List[Tuple[str, str, str]]:
        """데이터셋 생성"""
        dataset = []
        
        print(f"🔄 도메인9 데이터셋 {count}개 생성 중...")
        
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
        # 행동 유형 분포
        behavior_distribution = {bt: 0 for bt in self.behavior_types.keys()}
        
        # 측정 단위 분포
        unit_distribution = {"거리(m)": 0, "시간(초)": 0}
        
        # 기준 설정 분포
        baseline_distribution = {"기준있음": 0, "기준없음": 0}
        
        # 시간 형식 분포
        time_formats = {"HH:MM": 0, "한글시간": 0}
        
        # 문장 길이 분포
        sentence_lengths = {"2문장": 0, "3문장": 0, "기타": 0}
        
        # 상황 유형 분포
        situation_types = {"초과상황": 0, "정상상황": 0, "기준없음상황": 0}
        
        for input_str, output_str, domain in dataset:
            # 행동 유형 체크
            for behavior in self.behavior_types.keys():
                if behavior in input_str:
                    behavior_distribution[behavior] += 1
                    break
            
            # 측정 단위 체크
            if "m" in input_str and "분" not in input_str:
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
            
            # 문장 길이 체크 (소수점 제외하고 진짜 문장 종결만 카운트)
            # 소수점 마침표를 제외한 진짜 문장 종결어미만 찾기
            sentence_count = 0
            
            # 소수점이 아닌 마침표로 끝나는 문장들을 찾기
            # 숫자.숫자 패턴(소수점)을 제외하고 문장 종결 마침표만 카운트
            sentences = re.split(r'(?<!\d)\.(?!\d)', output_str)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = len(sentences)
            
            if sentence_count == 2:
                sentence_lengths["2문장"] += 1
            elif sentence_count == 3:
                sentence_lengths["3문장"] += 1
            else:
                sentence_lengths["기타"] += 1
            
            # 상황 유형 체크
            if any(word in output_str for word in ["초과", "넘어", "상회", "즉시"]):
                situation_types["초과상황"] += 1
            elif any(word in output_str for word in ["부합", "정상", "양호"]):
                situation_types["정상상황"] += 1
            else:
                situation_types["기준없음상황"] += 1
        
        total_count = len(dataset)
        
        return {
            "total_count": total_count,
            "behavior_distribution": {
                bt: f"{behavior_distribution[bt]} ({behavior_distribution[bt]/total_count*100:.1f}%)"
                for bt in self.behavior_types.keys()
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
            "sentence_lengths": {
                "2문장": f"{sentence_lengths['2문장']} ({sentence_lengths['2문장']/total_count*100:.1f}%)",
                "3문장": f"{sentence_lengths['3문장']} ({sentence_lengths['3문장']/total_count*100:.1f}%)",
                "기타": f"{sentence_lengths['기타']} ({sentence_lengths['기타']/total_count*100:.1f}%)"
            },
            "situation_types": {
                "초과상황": f"{situation_types['초과상황']} ({situation_types['초과상황']/total_count*100:.1f}%)",
                "정상상황": f"{situation_types['정상상황']} ({situation_types['정상상황']/total_count*100:.1f}%)",
                "기준없음상황": f"{situation_types['기준없음상황']} ({situation_types['기준없음상황']/total_count*100:.1f}%)"
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
    generator = Domain9QueueAlignmentGenerator()
    
    # 1000개 데이터셋 생성
    dataset = generator.generate_dataset(5000)
    
    # 저장 경로 설정
    output_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'domain9_dataset.csv')
    generator.save_to_csv(dataset, output_path)
    
    # 데이터셋 검증 및 결과 출력
    validation_results = generator.validate_dataset(dataset)
    
    print(f"\n✅ 도메인9 데이터셋 1000개가 생성되었습니다: {output_path}")
    print("\n📊 데이터셋 검증 결과:")
    print(f"   총 데이터 개수: {validation_results['total_count']}개")
    
    print(f"\n🎭 행동 유형 분포:")
    for behavior, stats in validation_results['behavior_distribution'].items():
        print(f"   {behavior}: {stats}")
    
    print(f"\n📏 측정 단위 분포:")
    print(f"   거리(m): {validation_results['unit_distribution']['거리(m)']}")
    print(f"   시간(초): {validation_results['unit_distribution']['시간(초)']}")
    
    print(f"\n📋 기준 설정 분포:")
    print(f"   기준있음: {validation_results['baseline_distribution']['기준있음']}")
    print(f"   기준없음: {validation_results['baseline_distribution']['기준없음']}")
    
    print(f"\n⏰ 시간 형식 분포:")
    print(f"   HH:MM 형식: {validation_results['time_formats']['HH:MM']}")
    print(f"   한글 시간: {validation_results['time_formats']['한글시간']}")
    
    print(f"\n📝 문장 길이 분포:")
    print(f"   2문장: {validation_results['sentence_lengths']['2문장']}")
    print(f"   3문장: {validation_results['sentence_lengths']['3문장']}")
    print(f"   기타: {validation_results['sentence_lengths']['기타']}")
    
    print(f"\n⚠️ 상황 유형 분포:")
    print(f"   초과상황: {validation_results['situation_types']['초과상황']}")
    print(f"   정상상황: {validation_results['situation_types']['정상상황']}")
    print(f"   기준없음상황: {validation_results['situation_types']['기준없음상황']}") 