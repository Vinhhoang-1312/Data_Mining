# MovieLens Data Mining — Unified Dashboard

Dự án phân tích và khai phá dữ liệu trên tập MovieLens, bao gồm:
- **Story A: Taste Tribes** — Phân cụm người dùng (K-Means Clustering) theo sở thích xem phim.
- **Story C: Behavioral Weirdness** — Phát hiện người dùng bất thường (Isolation Forest + LOF) và phim gây phân cực (Robust Z-Score).

**Author:** Vĩnh Hoàng | **Course:** Data Mining — MSA30DN, FSB  
**Instructor:** PhD. Cao Vu BUI

---

## 📂 Thư mục Artifacts dùng để làm gì?

Thư mục `artifacts/` đóng vai trò là **Kho lưu trữ kết quả đầu ra** (Output) của quá trình khai phá dữ liệu:
- **Tách biệt Dữ liệu & Kết quả**: Giúp mã nguồn sạch sẽ, không bị trộn lẫn với các file kết quả (json, html, png).
- **Lưu vết (Provenance)**: Chứa các file `run_manifest.json` để biết phiên bản code nào đã tạo ra kết quả này.
- **Phục vụ Dashboard**: Streamlit App sẽ đọc dữ liệu từ đây để hiển thị lên giao diện mà không cần chạy lại các phép toán nặng.

---

## 🏗️ Cấu trúc dự án (Project Structure)

```text
├── app.py                       # Entry point Dashboard (Streamlit) — tích hợp Story A + Story C
├── app_utils/                   # Thư mục chứa các mô-đun hỗ trợ (Modular Logic)
│   ├── config.py                # Quản lý đường dẫn, API Key và hằng số
│   ├── data_loader.py           # Xử lý đọc/ghi file Parquet và gọi TMDB API
│   ├── logic.py                 # Thuật toán tính toán User Vector & Phân cụm
│   ├── ui_components.py         # Giao diện Netflix CSS, Card, Grid và Skeleton Loading
│   └── visualizations.py        # Xử lý vẽ biểu đồ (Radar, t-SNE, PCA 3D)
├── story_a_taste_tribes.ipynb   # Notebook phân tích Story A (K-Means Clustering)
├── story_c_behavioral_weirdness.ipynb  # Notebook phân tích Story C (Anomaly Detection)
├── artifacts/                   # Kết quả đầu ra của Mining
│   ├── story_A/
│   │   ├── figures/             # Biểu đồ (HTML tương tác & PNG tĩnh)
│   │   ├── reports/             # File mô tả Cluster (Markdown), Manifest (JSON)
│   │   └── tables/              # Danh sách userId đã được gán nhãn cụm (Parquet)
│   └── story_C/
│       ├── figures/             # Biểu đồ Anomaly Scatter, Polarization Histogram
│       ├── reports/             # Summary, Case Studies, Run Manifest
│       └── tables/              # User Anomaly Scores, Movie Anomaly Scores (Parquet)
├── data-warehousing/            # Dữ liệu đầu vào đã qua tiền xử lý (Mining-Ready)
├── requirements.txt             # Danh sách thư viện cần thiết
└── .env                         # Chứa TMDB_API_KEY (Không đẩy lên Git)
```

## 📝 Giải thích chi tiết từng file

| File/Folder | Chức năng chính |
| :--- | :--- |
| `app.py` | Router chính — sidebar chọn Story A hoặc Story C, hiển thị Dashboard tương ứng. |
| `story_a_taste_tribes.ipynb` | Notebook chạy K-Means, tìm K tối ưu, trực quan hóa và xuất Artifacts cho Story A. |
| `story_c_behavioral_weirdness.ipynb` | Notebook chạy Isolation Forest + LOF + Robust Z-Score và xuất Artifacts cho Story C. |
| `app_utils/config.py` | Nơi duy nhất chứa các thông số cấu hình. Sửa ở đây sẽ áp dụng cho toàn dự án (DRY). |
| `app_utils/ui_components.py` | Chứa giao diện Netflix-style, đảm bảo UI chuyên nghiệp và nhất quán. |
| `app_utils/data_loader.py` | Đảm nhận việc lấy dữ liệu từ đĩa hoặc từ Internet (Poster phim qua TMDB API). |
| `app_utils/logic.py` | Xử lý các phép toán tìm cụm gần nhất khi người dùng chọn sở thích trên App. |
| `app_utils/visualizations.py` | Vẽ biểu đồ t-SNE 2D, PCA 3D với khả năng overlay vị trí người dùng mới. |
| `.gitignore` | Chặn các file rác (`__pycache__`), dữ liệu nặng và thông tin nhạy cảm. |

---

## 🚀 Cách chạy

### Bước 1: Chạy Notebooks để sinh Artifacts

Mở từng notebook trong VS Code (hoặc Jupyter) và chạy tất cả các cell theo thứ tự:
1. `story_a_taste_tribes.ipynb` → xuất artifacts vào `artifacts/story_A/`
2. `story_c_behavioral_weirdness.ipynb` → xuất artifacts vào `artifacts/story_C/`

### Bước 2: Chạy Dashboard

Mở terminal tại thư mục gốc của dự án và chạy:

```powershell
C:/ProgramData/anaconda3/envs/movielens/python.exe -m streamlit run app.py
```

### Tại sao dùng câu lệnh này?
1. **Đúng thư mục**: Chạy từ thư mục `project/` đảm bảo các import nội bộ (`app_utils`) hoạt động chính xác.
2. **Đúng môi trường**: Sử dụng trực tiếp file thực thi Python của môi trường `movielens`, tránh lỗi nếu lệnh `conda activate` chưa được cấu hình trong terminal của bạn.

---

## ✅ Kết quả kiểm thử (Verification)

- **Trạng thái**: Ứng dụng chạy tại `http://localhost:8501`.
- **Story A**: Dashboard tải thành công, hiển thị đầy đủ biểu đồ phân cụm (Radar, t-SNE, 3D PCA) và cho phép mô phỏng người dùng mới (Cold-Start Demo).
- **Story C**: Dashboard hiển thị User Anomaly Scatter, Polarization Histogram, bảng Top Anomalous Users, Top Polarizing Movies, và Reports.

---

*Dự án được thực hiện bởi **Vĩnh Hoàng** — Thành viên Team MovieLens Data Mining.*
