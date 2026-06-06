# Traffic Sign Recognition and Detection

## Giới thiệu

Đây là repository cho **báo cáo cuối kỳ môn học Nhập môn Xử lí ảnh số**.

Đề tài tập trung vào bài toán **phát hiện và nhận dạng biển báo giao thông** trong video bằng các kỹ thuật xử lí ảnh số truyền thống, kết hợp với phương pháp so khớp đặc trưng.

Hệ thống xử lý từng frame trong video, phát hiện các vùng có khả năng là biển báo dựa trên màu sắc và hình dạng, sau đó nhận dạng biển báo bằng ORB Feature Matching với các ảnh mẫu trong thư mục `template`.

## Thông tin môn học

- **Môn học:** Nhập môn Xử lí ảnh số
- **Loại báo cáo:** Báo cáo cuối kỳ
- **Đề tài:** Traffic Sign Recognition and Detection
- **Mã số sinh viên:** 52300135 - 52300138

## Nội dung repository

```text
TRAFFIC-SIGN-RECOGNITION-AND-DETECT/
│
├── 52300135_52300138.py      # Mã nguồn chính
├── 52300135_52300138.pdf     # Báo cáo cuối kỳ
├── OUTPUT_VIDEO.txt          # Đường dẫn / ghi chú video kết quả
├── template/                 # Ảnh mẫu dùng để nhận dạng biển báo
└── .gitattributes
```

## Mục tiêu đề tài

Đề tài hướng đến việc xây dựng một chương trình có khả năng:

- Đọc và xử lý video đầu vào.
- Phát hiện các vùng có khả năng chứa biển báo giao thông.
- Lọc đối tượng dựa trên màu sắc, hình dạng, diện tích và độ tròn.
- Nhận dạng biển báo bằng kỹ thuật ORB Feature Matching.
- Xuất video kết quả với bounding box và nhãn nhận dạng.

## Phương pháp thực hiện

Chương trình sử dụng các bước xử lý chính sau:

### 1. Đọc video đầu vào

Video được đọc bằng OpenCV thông qua `cv2.VideoCapture`.

```python
cap = cv2.VideoCapture('video2.mp4')
```

Sau khi xử lý, chương trình ghi kết quả ra file video mới:

```python
result = cv2.VideoWriter('video2_output.mp4', ...)
```

### 2. Chuyển đổi không gian màu

Mỗi frame được chuyển từ BGR sang HSV để thuận tiện cho việc tách màu biển báo.

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

### 3. Phát hiện biển báo bằng màu sắc

Chương trình tập trung vào các nhóm màu đặc trưng của biển báo, bao gồm:

- Màu đỏ
- Màu xanh

Các mask màu được tạo bằng `cv2.inRange`, sau đó xử lý bằng các phép toán hình thái học như:

- Dilation
- Opening
- Closing

### 4. Lọc contour theo hình dạng

Sau khi tìm contour, chương trình lọc các vùng nghi ngờ dựa trên:

- Diện tích contour
- Tỉ lệ khung bao `aspect ratio`
- Độ tròn `circularity`
- Mức độ lấp đầy của vùng mask

Hàm tính độ tròn:

```python
def circularity(cnt):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    if a < 1e-6 or p < 1e-6:
        return 0.0
    return float(4 * np.pi * a / (p * p))
```

### 5. Nhận dạng bằng ORB Feature Matching

Các ảnh mẫu trong thư mục `template` được đọc vào, chuyển sang ảnh xám, resize về kích thước chuẩn và trích xuất đặc trưng ORB.

```python
orb = cv2.ORB_create(nfeatures=1000)
```

Khi phát hiện được vùng nghi ngờ chứa biển báo, chương trình cắt ROI và so khớp đặc trưng với các template bằng BFMatcher.

```python
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

Nếu điểm so khớp vượt ngưỡng, biển báo được gán nhãn tương ứng.

## Công nghệ sử dụng

- Python
- OpenCV
- NumPy
- ORB Feature Detection
- BFMatcher
- HSV Color Segmentation
- Morphological Image Processing

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install opencv-python numpy
```

## Cách chạy chương trình

Đảm bảo trong thư mục project có video đầu vào tên:

```text
video2.mp4
```

Sau đó chạy:

```bash
python 52300135_52300138.py
```

Kết quả xử lý sẽ được lưu thành:

```text
video2_output.mp4
```

Trong quá trình chạy, chương trình cũng hiển thị cửa sổ kết quả với tên:

```text
EndTerm Output
```

Nhấn phím `q` để dừng chương trình.

## Cấu trúc thư mục template

Thư mục `template` chứa các ảnh mẫu dùng cho quá trình nhận dạng.

Mỗi loại biển báo nên được đặt trong một thư mục riêng, ví dụ:

```text
template/
│
├── stop/
│   ├── img1.jpg
│   └── img2.png
│
├── no_entry/
│   ├── img1.jpg
│   └── img2.png
│
└── turn_right/
    ├── img1.jpg
    └── img2.png
```

Tên thư mục sẽ được dùng làm nhãn nhận dạng cho biển báo.

## Kết quả

Chương trình có thể phát hiện các biển báo giao thông trong video dựa trên màu sắc và hình dạng, sau đó nhận dạng bằng cách so khớp với ảnh mẫu.

Kết quả đầu ra bao gồm:

- Khung bao quanh biển báo được phát hiện.
- Nhãn biển báo nếu nhận dạng thành công.
- Video kết quả sau xử lý.

## Báo cáo

Chi tiết quá trình thực hiện, cơ sở lý thuyết, thuật toán và kết quả được trình bày trong file báo cáo:

```text
52300135_52300138.pdf
```

## Ghi chú

- Chương trình sử dụng phương pháp xử lí ảnh truyền thống, không dùng mô hình học sâu.
- Độ chính xác phụ thuộc vào chất lượng video, điều kiện ánh sáng, góc nhìn và chất lượng ảnh mẫu trong thư mục `template`.
- Cần đảm bảo video đầu vào và thư mục template được đặt đúng vị trí trước khi chạy chương trình.

## Tác giả

- 52300135
- 52300138

## License

Repository này được xây dựng phục vụ mục đích học tập và báo cáo cuối kỳ cho môn học **Nhập môn Xử lí ảnh số**.
