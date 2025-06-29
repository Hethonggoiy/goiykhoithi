# goiykhoithi
# 🎓 Hệ thống Gợi ý Khối thi Đại học bằng Kỹ thuật phân lớp

**Dự án  hỗ trợ học sinh lớp 12 xác định khối thi đại học phù hợp nhất**, dựa trên học lực, sở thích và kết quả trắc nghiệm. Hệ thống sử dụng mô hình lai giữa MLP và XGBoost, triển khai trực quan trên giao diện web bằng Streamlit.

> 📍 Repo chính thức: https://github.com/Hethonggoiy/goiykhoithi  
> 🧠 Tác giả mô hình MLP: Bạn Truong Minh Diep– Sinh viên LHU 2025

---

## 🚀 Các tính năng chính

- Nhập điểm trung bình 10 môn + lựa chọn sở thích học tập
- Dự đoán **Top-3 khối thi phù hợp** (A00, B00, C00, D01…)
- Gợi ý ngành học tương ứng với khối thi
- Trực quan hóa xác suất bằng biểu đồ
- Dự đoán hàng loạt từ file Excel (.xlsx)
- Ghi log vào hệ thống cơ sở dữ liệu SQLite3
- Có thể huấn luyện lại mô hình từ file `.csv` người dùng

---

## 📦 Cấu trúc thư mục

goiykhoithi/ ├── web_app.py                  
# Ứng dụng Streamlit chính ├── train_hybrid.py           
# Huấn luyện mô hình lai MLP + XGB ├── export_excel.py            
# Trích xuất dữ liệu log sang Excel ├── requirements.txt          
# Gói pip cần cài đặt ├── README.md                  
# Tài liệu hướng dẫn ├── db/ │   └── goiy_khoithi.db        
# CSDL SQLite3 ghi log ├── models/ │   ├── mlp_model.h5 │   ├── xgb_model.pkl │   ├── scaler.pkl │   └── label_encoder.pkl └── data/ ├── du_lieu_chuan.csv      
# Dữ liệu mẫu để huấn luyện ├── Khoi-thi.xlsx          # Tổ hợp môn → khối thi └── khoi_nganh.csv         # Khối thi → ngành học phù hợp
