import cv2
import torch
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Config value
video_path = "C:/Users/Duong/PycharmProjects/tracking_obj/traffic_violations/data/1.mp4"
conf_threshold = 0.5
tracking_class = [2, 3, 5, 7, 9]

# Khởi tạo DeepSort
tracker = DeepSort(max_age=15)

# Khởi tạo YOLOv9
device = "cpu"
model  = DetectMultiBackend(weights="C:/Users/Duong/PycharmProjects/tracking_obj/traffic_violations/yolov9/weights/yolov9-c.pt", device=device, fuse=True )
model  = AutoShape(model)

# Load classname từ file classes.names
with open("C:/Users/Duong/PycharmProjects/tracking_obj/traffic_violations/data/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names),3 ))
tracks = []

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

#tao thu muc luu anh crop
crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

# Tiến hành đọc từng frame từ video
idx = 0  # Để đảm bảo tên file là duy nhất
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Thoát khỏi vòng lặp nếu không còn frame nào để đọc

    # Dự đoán các đối tượng trong frame sử dụng mô hình
    results = model(frame)
    detect = []

    for detect_obj in results.pred[0]:
        label, confidence, bbox = detect_obj[5], detect_obj[4], detect_obj[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        # Kiểm tra xem đối tượng có nằm trong lớp được theo dõi không và đạt độ tin cậy không
        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id not in tracking_class or confidence < conf_threshold:
                continue

        # Nếu đối tượng là lớp 9, cắt và lưu ảnh
        if class_id == 9:
            crop = frame[y1:y2, x1:x2]  # Chú ý thứ tự của slicing
            cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), crop)
            idx += 1  # Tăng số thứ tự file ảnh lên sau mỗi lần lưu

        # Thêm thông tin của đối tượng vào danh sách detect
        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Cập nhật,gán ID bằng DeepSort
    tracks = tracker.update_tracks(detect, frame = frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int,color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()