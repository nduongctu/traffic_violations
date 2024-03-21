import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading

class VideoPlayer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Player")
        self.geometry("1100x900")

        # Khung chứa video
        self.video_frame = tk.Frame(self, width=800, height=600)
        self.video_frame.pack(side=tk.LEFT, anchor=tk.NW)

        # Nhãn hiển thị video
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Khung chứa các nút bên phải video
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.RIGHT, anchor=tk.NE, padx=20, pady=20)

        # Nút tải lên video
        self.upload_button = tk.Button(self.button_frame, text="Tải lên video", command=self.open_file)
        self.upload_button.pack(pady=10)

        # Nút vẽ đường thẳng
        self.draw_line_button = tk.Button(self.button_frame, text="Vẽ đường thẳng", command=self.draw_line)
        self.draw_line_button.pack(pady=10)

        # Nút xử lý
        self.process_button = tk.Button(self.button_frame, text="Xử lý", command=self.process)
        self.process_button.pack(pady=10)

        self.video_path = None
        self.video_thread = None
        self.line_start = None
        self.line_end = None

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi;*.mov")])
        if file_path:
            self.video_path = file_path
            self.start_video()

    def start_video(self):
        if self.video_thread is not None:
            self.stop_video()

        self.video_thread = threading.Thread(target=self.update_video, args=(), daemon=True)
        self.video_thread.start()

    def update_video(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to fixed size 800x600
            frame = cv2.resize(frame, (900, 700))

            # Vẽ đường thẳng nếu đã được xác định
            if self.line_start is not None and self.line_end is not None:
                cv2.line(frame, self.line_start, self.line_end, (0, 0, 255), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(image)

            self.video_label.configure(image=photo)
            self.video_label.image = photo  # keep a reference to prevent garbage collection
            self.video_label.update()

        cap.release()

    def stop_video(self):
        if self.video_thread is not None:
            self.video_thread.join()
            self.video_thread = None

    def draw_line(self):
        # Bắt đầu vẽ đường thẳng khi nút "Vẽ đường thẳng" được bấm
        self.video_label.bind("<Button-1>", self.start_drawing_line)
        self.video_label.bind("<B1-Motion>", self.update_drawing_line)
        self.video_label.bind("<ButtonRelease-1>", self.finish_drawing_line)

    def start_drawing_line(self, event):
        # Lưu tọa độ điểm bắt đầu vẽ đường thẳng
        self.line_start = (event.x, event.y)

    def update_drawing_line(self, event):
        # Cập nhật tọa độ điểm kết thúc vẽ đường thẳng
        self.line_end = (event.x, event.y)

    def finish_drawing_line(self, event):
        # Kết thúc vẽ đường thẳng
        self.line_end = (event.x, event.y)
        self.video_label.unbind("<Button-1>")
        self.video_label.unbind("<B1-Motion>")
        self.video_label.unbind("<ButtonRelease-1>")
        with open('toado.txt', 'w') as file:
            file.write(f'x1={self.line_start[0]},y1={self.line_start[1]}\n')
            file.write(f'x2={self.line_end[0]},y2={self.line_end[1]}')

    def process(self):
        pass  # Hàm xử lý khi bấm nút "Xử lý"

if __name__ == "__main__":
    app = VideoPlayer()
    app.mainloop()
