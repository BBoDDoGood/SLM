# 개선된 프롬프트 형식
def create_improved_input(domain, input_text):
    return f"""다음은 한국어로 답변해주세요.

상황: {input_text}
도메인: {domain}

위 상황을 분석하고 한국어로 적절한 대응 방안을 제시해주세요:"""

# 예시
original_input = "도메인: 군중 밀집 및 체류 감지 입력: 14:10 전시관 1층 A입구 앞 감지 인원 62명, 기준 인원 40명"

improved_input = """다음은 한국어로 답변해주세요.

상황: 14:10 전시관 1층 A입구 앞 감지 인원 62명, 기준 인원 40명
도메인: 군중 밀집 및 체류 감지

위 상황을 분석하고 한국어로 적절한 대응 방안을 제시해주세요:"""

print("기존 입력:")
print(original_input)
print("\n개선된 입력:")
print(improved_input)
