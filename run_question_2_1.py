from solution.solution import chat_with_system_prompt


user_prompt = "Giải thích máy học (machine learning) là gì?"

personas = [
    (
        "NHÀ THƠ",
        "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ.",
    ),
    (
        "KỸ SƯ PHẦN MỀM SENIOR",
        "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp.",
    ),
]

results = []

for name, system_prompt in personas:
    answer, latency = chat_with_system_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=350,
    )

    result = (
        f"\n{'=' * 70}\n"
        f"PERSONA: {name}\n"
        f"{'=' * 70}\n"
        f"{answer}\n"
        f"Độ trễ: {latency:.2f} giây\n"
    )

    print(result)
    results.append(result)

with open("results_question_2_1.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(results))

print("\nĐã lưu kết quả vào results_question_2_1.txt")