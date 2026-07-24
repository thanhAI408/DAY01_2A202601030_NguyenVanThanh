from solution.solution import call_openai


prompt = "Hãy kể cho tôi một sự thật thú vị về Hà Nội."
temperatures = [0.0, 0.7, 1.2, 1.8]

results = []

for temperature in temperatures:
    answer, latency = call_openai(
        prompt=prompt,
        temperature=temperature,
        top_p=0.9,
        max_tokens=200,
    )

    result = (
        f"\n{'=' * 60}\n"
        f"TEMPERATURE = {temperature}\n"
        f"{'=' * 60}\n"
        f"{answer}\n"
        f"Độ trễ: {latency:.2f} giây\n"
    )

    print(result)
    results.append(result)

# Lưu kết quả để dễ xem lại
with open("results_question_1_1.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(results))

print("\nĐã lưu kết quả vào results_question_1_1.txt")