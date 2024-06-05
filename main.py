import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas, NW, messagebox
import cv2
from PIL import Image, ImageTk
import os

# Path to the MobileNet SSD model files
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Load the object detection model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
classLabels = []
file_name = 'Labels.txt'  # Ensure this file contains COCO class labels
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

class Application:
    def __init__(self, window):
        self.window = window
        self.window.title("Object Detection GUI")
        self.window.geometry("800x600")
        self.window.configure(background='#ffffff')

        self.canvas = Canvas(window, width=640, height=480, bg='white')
        self.canvas.pack(pady=20)

        self.btn_frame = Frame(window, bg='#ffffff')
        self.btn_frame.pack()

        self.btn_select_image = Button(self.btn_frame, text="Select Image", command=self.select_image, bg='#4CAF50', fg='white', padx=10, pady=5)
        self.btn_select_image.grid(row=0, column=0, padx=10, pady=10)

        self.btn_live_feed = Button(self.btn_frame, text="Live Feed", command=self.live_feed, bg='#4CAF50', fg='white', padx=10, pady=5)
        self.btn_live_feed.grid(row=0, column=1, padx=10, pady=10)

        self.label_status = Label(window, text="Select an image or start the live feed.", bg='#ffffff', fg='#000000', pady=10)
        self.label_status.pack()

        self.cap = None
        self.is_live_feed_running = False

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if not os.path.exists(file_path):
                messagebox.showerror("Error", "File not found. Please check the file path.")
                return
            self.label_status.config(text="Processing image...")
            self.process_image(file_path)

    def process_image(self, file_path):
        # Debugging: Print the file path
        print(f"Selected file path: {file_path}")

        # Verify file exists
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", "File not found or inaccessible. Please check the file path.")
            self.label_status.config(text="Failed to read the image file.")
            return

        # Verify file extension
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            messagebox.showerror("Error", "Unsupported file format. Please select a valid image file.")
            self.label_status.config(text="Unsupported file format.")
            return

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Failed to read the image file. Please check the file integrity.")
            self.label_status.config(text="Failed to read the image file.")
            return

        self.detect_and_display(image)
        self.label_status.config(text="Image processed. Select another image or start the live feed.")

    def live_feed(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label_status.config(text="Failed to access webcam.")
            return
        self.label_status.config(text="Press 'q' to exit live feed.")
        self.is_live_feed_running = True
        self.update_frame()

    def update_frame(self):
        if self.is_live_feed_running:
            ret, frame = self.cap.read()
            if ret:
                self.detect_and_display(frame)
                self.window.after(10, self.update_frame)
            else:
                self.cap.release()
                self.label_status.config(text="Live feed stopped.")

    def detect_and_display(self, image):
        if image is None or image.size == 0:
            messagebox.showerror("Error", "Invalid image. Please try again.")
            return

        # Perform the detection
        ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.55)

        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd <= 80:
                    cv2.rectangle(image, boxes, (255, 0, 0), 2)
                    cv2.putText(image, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
        self.canvas.image = imgtk

    def on_key_press(self, event):
        if event.char == 'q':
            self.is_live_feed_running = False
            if self.cap:
                self.cap.release()
            self.label_status.config(text="Live feed stopped.")
            self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.bind('<KeyPress>', app.on_key_press)
    root.mainloop()
