import cv2
import numpy as np
import os
import glob

def circularity(cnt):
    """
    Tính toán độ tròn (circularity). 1.0 là hình tròn hoàn hảo.
    """
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    if a < 1e-6 or p < 1e-6: return 0.0
    return float(4 * np.pi * a / (p * p))

def load_templates(template_dir="template"):
    """
    Load template 
    """
    templates_orb = {}
    orb = cv2.ORB_create(nfeatures=1000)

    if not os.path.exists(template_dir):
        print(f"Warning: Template directory '{template_dir}' not found!")
        return templates_orb

    for label_folder in os.listdir(template_dir):
        folder_path = os.path.join(template_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        label = label_folder
        image_files = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if len(image_files) == 0:
            print(f"Warning: Folder '{label}' does not contain any images.")
            continue

        kp_list = []
        des_list = []
        img_list = []

        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Cannot read {img_path}")
                continue

            img = cv2.resize(img, (100, 100))
            img = cv2.equalizeHist(img)

            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                kp_list.append(kp)
                des_list.append(des)
                img_list.append(img)
        if len(des_list) == 0:
            print(f"Warning: No features found in any images of '{label}'")
            continue
        templates_orb[label] = {
            "imgs": img_list,
            "kp": kp_list,
            "des": des_list,
            "label": label
        }
        print(f"Loaded template: {label} - {sum(len(k) for k in kp_list)} total keypoints ({len(kp_list)} images)")
    print(f"Total templates loaded: {len(templates_orb)}")
    return templates_orb

def orb_classify(roi_bgr, templates_orb, orb_detector, bf_matcher):
    """
    Phân loại ROI bằng ORB Feature Matching.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "Unknown", 0.0

    # chuyển đổi ROI sang ảnh xám
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (100, 100))
    roi_gray = cv2.equalizeHist(roi_gray)

    # Tìm keypoint + descriptor của ROI
    kp_roi, des_roi = orb_detector.detectAndCompute(roi_gray, None)

    # Nếu không đủ đặc trưng thì bỏ
    if des_roi is None or len(des_roi) < 1:
        return "Unknown", 0.0

    best_label = "Unknown"
    best_score = 0.0
    threshold = 0.30

    # ---- Lặp qua từng template để so khớp ----
    for label, tpl in templates_orb.items():
        des_list = tpl.get('des', [])
        kp_list = tpl.get('kp', [])
        label_best = 0.0
        for des_tpl, kp_tpl in zip(des_list, kp_list):
            try:
                matches = bf_matcher.match(des_tpl, des_roi)
                good_matches = [m for m in matches if m.distance < 70]
                score = len(good_matches) / (len(kp_tpl) + 1e-6)
                if score > label_best:
                    label_best = score
            except Exception:
                continue
        if label_best > best_score:
            best_score = label_best
            best_label = label

    # Không đủ match thì đặt label là Unknown
    if best_score < threshold:
        return "Unknown", best_score

    return best_label, best_score

def main():
    # Khởi tạo Video I/O 
    cap = cv2.VideoCapture('video2.mp4')
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    result = cv2.VideoWriter('video2_output.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, size)

    # Khởi tạo ORB và Matcher
    orb_detector = cv2.ORB_create(nfeatures=1500)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Load Templates 
    templates_orb = load_templates("template")

    use_feature_matching = len(templates_orb) > 0

    # Dải màu HSV 
    color_ranges = {
        "Red": [
            (np.array([0, 40, 20]),   np.array([12, 255, 255])),  
            (np.array([160, 40, 20]), np.array([179, 255, 255]))
        ],
        "Blue": [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
    }

    # Kernels
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Vòng lặp xử lý
    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        cv2.putText(frame, "52300135_52300138", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if not ret:
            print("End of video.")
            break

        """Frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        vis = frame.copy() 
        detections = []
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        """Xử lí màu"""
        for color, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lo, hi) in ranges:
                mask |= cv2.inRange(hsv, lo, hi)

            """
            Morphology
            Phần xử lý được xử lý riêng màu đỏ và màu xanh vì các đặc trưng riêng 
            """
            if color == "Blue":
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, 1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16)), 2)
                mask = cv2.dilate(mask, k5, iterations=1)
            elif color == "Red": #đỏ
                mask = cv2.dilate(mask, k7, 2)               
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, 2)  

            mask_total = cv2.bitwise_or(mask_total, mask)  
            
            # Tìm contour trên mask 
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not cnts:
                continue

            for c in cnts:
                area = cv2.contourArea(c)
                if area < 1500: 
                    continue

                # Lọc Aspect Ratio 
                x, y, w_rect, h_rect = cv2.boundingRect(c)
                ar = w_rect / float(h_rect)
                if ar < 0.6 or ar > 1.5: 
                    continue

                circ = circularity(c)
                
                if color == "Red":
                    if circ > 0.70:
                        roi = mask[y:y + h_rect, x:x + w_rect]
                        fill = roi.mean() / 255.0
                        if fill > 0.01:
                            detections.append((x, y, w_rect, h_rect, color))

                    elif circ > 0.50:
                        roi = mask[y:y + h_rect, x:x + w_rect]
                        fill = roi.mean() / 255.0
                        # Bỏ vật thể đỏ đặc (fill cao) != hình tam giác 
                        if fill > 0.50:
                            continue
                        
                        # Giữ biển báo đỏ: viền đỏ, ruột vàng nên fill vừa phải 
                        if 0.05 < fill < 0.55:
                            detections.append((x, y, w_rect, h_rect, color))
                
                elif color == "Blue":
                    # chỉ kiểm tra hình tròn cho biển xanh
                    if circ > 0.70:
                        roi = mask[y:y + h_rect, x:x + w_rect]
                        fill = roi.mean() / 255.0
                        if fill > 0.35:
                            detections.append((x, y, w_rect, h_rect, color))

        # FEATURE MATCHING (Nếu có templates) 
        recognized_signs = []
        if use_feature_matching:
            for (x, y, w_rect, h_rect, color) in detections:
                # Cắt ROI từ frame gốc (thêm padding nhỏ)
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame_width, x + w_rect + padding)
                y2 = min(frame_height, y + h_rect + padding)
                
                roi_bgr = frame[y1:y2, x1:x2]
                
                if roi_bgr.size == 0:
                    continue
                    
                # Phân loại bằng feature matching
                label, score = orb_classify(roi_bgr, templates_orb, orb_detector, bf_matcher)
                
                if label != "Unknown":
                    recognized_signs.append((x, y, w_rect, h_rect, color, label, score))
        else:
            pass

        recognized_positions = {(x, y, w, h) for (x, y, w, h, _, _, _) in recognized_signs}

        # Biển báo bị phát hiện nhưng không được nhận dạng        
        for (x, y, w, h, color) in detections:
            if (x, y, w, h) not in recognized_positions:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)  
                cv2.putText(vis, "", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
        # Vẽ đối tượng đã nhận dạng được
        for (x, y, w, h, color, label, score) in recognized_signs:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)  # xanh lá
            display_label = label
            cv2.putText(vis, f"{display_label}", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        result.write(vis)
        cv2.imshow('EndTerm Output', vis)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    print("Done. Releasing resources.")
    cap.release()
    result.release()
    cv2.destroyAllWindows()

# Chạy chương trình
if __name__ == "__main__":
    main()