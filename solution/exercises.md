# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Qua bốn lần gọi API, khi temperature tăng, nội dung phản hồi có xu hướng đa dạng hơn về chủ đề và cách trình bày: từ 36 phố phường, Hồ Hoàn Kiếm, phố đường tàu đến lễ kỷ niệm 1000 năm Thăng Long – Hà Nội. Trong lần thử này, cả bốn phản hồi vẫn khá mạch lạc; tuy nhiên ở temperature 1.2 bắt đầu xuất hiện một vài lỗi định dạng như các từ bị dính liền, cho thấy đầu ra có phần kém ổn định hơn. Temperature 1.8 vẫn rõ ràng trong lần chạy này nên chưa thể kết luận rằng mức này luôn tạo phản hồi kém mạch lạc.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn thảo hợp đồng pháp lý, tôi chọn temperature khoảng 0.1 vì công việc này cần phản hồi ổn định, chính xác và hạn chế model tự sáng tạo thêm nội dung. Với trợ lý viết slogan quảng cáo, tôi chọn temperature khoảng 1.0 để tạo ra nhiều cách diễn đạt mới lạ và đa dạng hơn. Sự khác biệt là hợp đồng pháp lý ưu tiên tính nhất quán, còn slogan quảng cáo ưu tiên tính sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
>  Mỗi ngày có 20.000 × 2 = 40.000 lượt gọi API. Với trung bình 500 token đầu ra mỗi lượt, tổng số token đầu ra là 40.000 × 500 = 20.000.000 token/ngày. Theo bảng giá trong template, GPT-4o có chi phí khoảng 20.000.000/1.000 × 0,010 = 200 USD/ngày, còn GPT-4o-mini có chi phí khoảng 20.000.000/1.000 × 0,0006 = 12 USD/ngày. Model lớn phù hợp với các tác vụ phức tạp như phân tích tài liệu pháp lý hoặc lập luận nhiều bước, còn model nhỏ phù hợp với chatbot hỗ trợ khách hàng hoặc các tác vụ đơn giản có số lượng người dùng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona nhà thơ, mô hình trả lời ngắn gọn, giàu hình ảnh ví von như “hạt giống dữ liệu” và “cây kiến thức”, đồng thời gần như không sử dụng thuật ngữ chuyên môn. Với persona kỹ sư phần mềm senior, phản hồi dài và có cấu trúc hơn, giải thích các loại máy học bằng thuật ngữ kỹ thuật như supervised learning, classification và regression, đồng thời cung cấp ví dụ Python. Qua đó, system prompt có thể điều khiển rõ rệt giọng văn, độ dài, mức độ kỹ thuật, cách tổ chức nội dung và việc có đưa ví dụ code hay không.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Đoạn văn tôi sử dụng có 160 từ. Hàm count_tokens đếm được 200 token, trong khi cách ước lượng số từ/0.75 cho kết quả khoảng 213,33 token; hai kết quả chênh nhau khoảng 6,67%. Trong trường hợp này, cách ước lượng thô cho số token cao hơn kết quả tiktoken nên sẽ làm dự toán ngân sách API bị thừa. Nguyên nhân là mối quan hệ giữa số từ và số token không cố định mà còn phụ thuộc vào ngôn ngữ, cách viết và bộ mã hóa của model.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản hưởng lợi nhiều từ streaming vì người dùng có thể bắt đầu đọc câu trả lời ngay khi model đang sinh nội dung, thay vì phải chờ toàn bộ phản hồi hoàn thành. Trợ lý giọng nói cũng hưởng lợi vì có thể bắt đầu đọc sớm, tuy nhiên cần có bộ đệm để tránh đọc các câu chưa hoàn chỉnh. Pipeline dịch tài liệu chạy ngầm ban đêm ít cần streaming nhất vì người dùng chỉ quan tâm đến bản dịch cuối cùng, không cần theo dõi kết quả xuất hiện theo thời gian thực.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Khi API quá tải, exponential backoff làm thời gian chờ giữa các lần thử lại tăng dần, ví dụ 0,1 giây, 0,2 giây rồi 0,4 giây. Cách này giúp giảm số request liên tục gửi tới server và cho server thêm thời gian phục hồi, hiệu quả hơn delay cố định. Tuy nhiên, nếu nhiều client cùng bắt đầu retry một lúc, chúng vẫn có thể gửi lại request tại các thời điểm giống nhau; jitter thêm một khoảng trễ ngẫu nhiên để phân tán các request và tránh hiện tượng tất cả client cùng retry đồng thời.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt tôi sử dụng là: “Bạn là trợ giảng AI thân thiện. Hãy trả lời ngắn gọn bằng tiếng Việt, giải thích từng bước và đưa ra ví dụ Python đơn giản khi phù hợp.” Nếu xóa cụm “trả lời ngắn gọn bằng tiếng Việt”, trợ lý có thể trả lời dài dòng hoặc sử dụng ngôn ngữ khác. Nếu xóa cụm “giải thích từng bước và đưa ra ví dụ Python đơn giản”, phản hồi có thể chỉ trình bày lý thuyết chung mà không hướng dẫn rõ quy trình hoặc minh họa bằng code.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
>  Giả sử ở lượt đầu người dùng yêu cầu xây dựng một chatbot bằng Python với điều kiện không sử dụng framework, sau đó hai bên trao đổi hơn bốn lượt về giao diện, dữ liệu và cách triển khai. Khi người dùng nói “hãy sửa code theo yêu cầu ban đầu”, thông tin ở lượt đầu có thể đã bị loại khỏi history nên trợ lý không còn nhớ điều kiện không sử dụng framework và đưa ra giải pháp sai. Cách khắc phục là tóm tắt các lượt cũ thành một phần memory ngắn, luôn gửi phần tóm tắt đó cùng bốn lượt gần nhất, hoặc giữ chọn lọc các yêu cầu quan trọng thay vì chỉ cắt theo số lượng message.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
