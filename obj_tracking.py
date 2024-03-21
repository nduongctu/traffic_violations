import cv2
import torch
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from collections import deque

# Định nghĩa hàm detect_traffic_light_color
def detect_traffic_light_color(image, rect):
    # Trích xuất kích thước hình chữ nhật
    x, y, w, h = rect
    # Trích xuất vùng quan tâm (ROI) từ ảnh dựa trên hình chữ nhật
    roi = image[y:y + h, x:x + w]

    # Chuyển ROI sang không gian màu HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Định nghĩa phạm vi HSV cho màu đỏ
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])

    # Định nghĩa phạm vi HSV cho màu vàng
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Tạo mask nhị phân để phát hiện màu đỏ và màu vàng trong ROI
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Thông tin về font để đặt chữ lên ảnh
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    font_thickness = 2

    # Kiểm tra màu nào hiện diện dựa trên các mask
    if cv2.countNonZero(red_mask) > 0:
        text_color = (0, 0, 255)
        message = "Trang thai tin hieu: Do"
        color = 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color = (0, 255, 255)
        message = "Trang thai tin hieu: Vang"
        color = 'yellow'
    else:
        text_color = (0, 255, 0)
        message = "Trang thai tin hieu : Xanh"
        color = 'green'

    # Chồng lớp trạng thái đè lên ảnh gốc
    cv2.putText(image, message, (15, 70), font, font_scale - 5, text_color, font_thickness + 1, cv2.LINE_AA)
    # Trả về ảnh đã chỉnh sửa và màu được phát hiện
    return image, color

# Các giá trị cấu hình
video_path = "C://Users/Duong/PycharmProjects/nienluannganh/traffic_violations/data/1.mp4"
conf_threshold = 0.25
tracking_class = [2, 3, 5, 7, 9]

# Khởi tạo DeepSort
tracker = DeepSort(max_age=25)

# Khởi tạo YOLOv9
device = "cpu"
model = DetectMultiBackend(
    weights="C:/Users/Duong/PycharmProjects/nienluannganh/traffic_violations/yolov9/weights/yolov9-c.pt", device=device,
    fuse=True)
model = AutoShape(model)

# Load tên các lớp từ tệp classes.names
with open("C:/Users/Duong/PycharmProjects/tracking_obj/traffic_violations/data/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Khởi tạo VideoCapture để đọc từ tệp video
cap = cv2.VideoCapture(video_path)
new_width = 900
new_height = 700

# Tạo thư mục để lưu các ảnh vi phạm
violation_dir_name = "violations"
if not os.path.exists(violation_dir_name):
    os.mkdir(violation_dir_name)


# Lấy thông tin về video (kích thước khung hình, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo VideoWriter để ghi video kết quả
output_video_path = "result/result_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

# Định nghĩa hai điểm đầu và cuối của đường thẳng
def read_coordinates_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            coordinates = {}
            for line in lines:
                parts = line.strip().split(",")
                for part in parts:
                    key, value = part.split("=")
                    coordinates[key] = int(value)
            # Kiểm tra xem các giá trị cần thiết đã được đọc chưa
            if 'x1' in coordinates and 'y1' in coordinates and 'x2' in coordinates and 'y2' in coordinates:
                return (coordinates['x1'], coordinates['y1']), (coordinates['x2'], coordinates['y2'])
            else:
                print("Invalid coordinates in file. Make sure x1, y1, x2, y2 are all present.")
                return None, None
    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
        return None, None


# Đường dẫn đến file toado.txt
file_path = "toado.txt"

# Đọc tọa độ từ file
point1, point2 = read_coordinates_from_file(file_path)

# In ra tọa độ
print("point1:", point1)
print("point2:", point2)

def xe_vuot_duong_thang(x1, y1, x2, y2, point1, point2):
    """
    Kiểm tra xem hình chữ nhật (x1, y1, x2, y2) có vượt qua đường thẳng (point1, point2) hay không.

    Args:
        x1 (int): Tọa độ x của góc trên bên trái hình chữ nhật.
        y1 (int): Tọa độ y của góc trên bên trái hình chữ nhật.
        x2 (int): Tọa độ x của góc dưới bên phải hình chữ nhật.
        y2 (int): Tọa độ y của góc dưới bên phải hình chữ nhật.
        point1 (tuple): Tọa độ (x, y) của điểm đầu tiên của đường thẳng.
        point2 (tuple): Tọa độ (x, y) của điểm thứ hai của đường thẳng.

    Returns:
        bool: True nếu hình chữ nhật vượt qua đường thẳng, False nếu ngược lại.
    """
    # Tính hệ số a, b của phương trình đường thẳng y = ax + b
    a = (point2[1] - point1[1]) / (point2[0] - point1[0] + 1e-9)  # Tránh chia cho 0
    b = point1[1] - a * point1[0]

    # Kiểm tra xem bốn góc của hình chữ nhật có nằm hai phía khác nhau của đường thẳng hay không
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    above = below = False
    for px, py in points:
        y_on_line = a * px + b
        if py > y_on_line:
            above = True
        else:
            below = True

    if above and below:
        return True
    else:
        return False

idx = 0

while True:
    ret, frame = cap.read()
    
    if frame is None:  # Kiểm tra xem frame có tồn tại không
        break

    frame_resized = cv2.resize(frame, (new_width, new_height))

    if point1 is not None and point2 is not None:
        a = (point2[1] - point1[1]) / (point2[0] - point1[0] + 1e-9)  # Tránh chia cho 0
        b = point1[1] - a * point1[0]

        # Thay đổi màu đường thẳng hiện tại thành đỏ
        cv2.line(frame_resized, point1, point2, (0, 0, 255), 2)

        # Vẽ đường thẳng song song phía dưới đường đỏ, cách đường đỏ một khoảng cách bằng 100 pixel
        distance = 200  # Khoảng cách song song
        new_b = b + distance  # Thay đổi hệ số b để tạo đường thẳng song song

        # Tính tọa độ điểm đầu và điểm cuối của đường thẳng mới
        new_x1 = 0
        new_y1 = int(a * new_x1 + new_b)
        new_x2 = frame_resized.shape[1] - 1
        new_y2 = int(a * new_x2 + new_b)

        new_point1 = (new_x1, new_y1)
        new_point2 = (new_x2, new_y2)
        cv2.line(frame_resized, new_point1, new_point2, (0, 255, 0), 2)

    results = model(frame_resized)
    detect = []

    for detect_obj in results.pred[0]:
        label, confidence, bbox = detect_obj[5], detect_obj[4], detect_obj[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id not in tracking_class or confidence < conf_threshold:
                continue

        if class_id == 9:  # Kiểm tra xem đối tượng được phát hiện có phải là đèn giao thông không
            # Gọi hàm detect_traffic_light_color để phát hiện đèn giao thông
            frame_resized, color = detect_traffic_light_color(frame_resized, (x1, y1, x2 - x1, y2 - y1))
            idx += 1

        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    tracks = tracker.update_tracks(detect, frame=frame_resized)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame_resized, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame_resized, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Kiểm tra xem xe có vượt đường thẳng hay không khi đèn đỏ
            if class_id in [2, 3, 5, 7] and xe_vuot_duong_thang(x1, y1, x2, y2, point1, point2):
                if (color == [0, 0, 255]).all():
                    video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # Lấy thời gian hiện tại trong video (milli giây)
                    video_time_sec = video_time_ms / 1000  # Chuyển đổi sang giây
                    frame_name = f"violation_{int(video_time_sec)}.png"
                    cv2.imwrite(os.path.join(violation_dir_name, frame_name), frame_resized)
                    print(f"Thời gian vượt đèn đỏ: {video_time_sec} giây")

    out.write(frame_resized)

    cv2.imshow("Resized Frame", frame_resized)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
