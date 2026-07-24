from solution.solution import count_tokens


text = """
Trí tuệ nhân tạo đang được ứng dụng rộng rãi trong giáo dục. Sinh viên có
thể dùng AI để tìm kiếm thông tin, giải thích khái niệm khó, kiểm tra lỗi
lập trình và xây dựng kế hoạch học tập. Tuy nhiên, AI không nên thay thế
hoàn toàn quá trình tư duy của người học. Nếu chỉ sao chép câu trả lời do
máy tạo ra, sinh viên có thể hoàn thành bài nhanh hơn nhưng không thực sự
hiểu kiến thức. Vì vậy, người học cần kiểm tra độ chính xác của thông tin,
đối chiếu với tài liệu đáng tin cậy và tự trình bày lại bằng cách hiểu của
mình. Giảng viên cũng cần hướng dẫn cách sử dụng AI có trách nhiệm, bảo vệ
dữ liệu cá nhân và ghi nhận sự hỗ trợ của công cụ. Nếu được dùng đúng cách,
AI có thể trở thành một trợ lý học tập hữu ích.
""".strip()

word_count = len(text.split())
actual_tokens = count_tokens(text)
estimated_tokens = word_count / 0.75

difference_percent = (
    abs(actual_tokens - estimated_tokens) / actual_tokens * 100
)

print(f"Số từ: {word_count}")
print(f"Token theo count_tokens: {actual_tokens}")
print(f"Ước lượng số từ / 0.75: {estimated_tokens:.2f}")
print(f"Chênh lệch: {difference_percent:.2f}%")

if estimated_tokens < actual_tokens:
    print("Kết luận: Cách ước lượng thô làm dự toán thiếu.")
else:
    print("Kết luận: Cách ước lượng thô làm dự toán thừa.")