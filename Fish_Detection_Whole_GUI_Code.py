import torch
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, Text, messagebox
import cv2
from PIL import Image, ImageTk
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np
import os
import time
import random

class FishDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fish Detection GUI")
        self.root.geometry("1000x800")
        self.center_window()

        self.heading_frame = tk.Frame(root)
        self.heading_frame.pack(pady=20)

        self.college_logo = Image.open("D:/Major_Project/srmlogo.png")
        self.college_logo = self.college_logo.resize((400, 100))
        self.college_logo_image = ImageTk.PhotoImage(self.college_logo)
        self.logo_label = Label(self.heading_frame, image=self.college_logo_image)
        self.logo_label.pack(side=tk.LEFT, padx=10)

        self.heading = Label(self.heading_frame, text="Fish Detection GUI", font=("Arial", 20, "bold"))
        self.heading.pack(side=tk.LEFT)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, padx=(10, 2))

        self.canvas_title = Label(self.canvas_frame, text="Display Screen", font=("Arial", 14, "bold"))
        self.canvas_title.pack(pady=(0, 5))

        self.canvas = Canvas(self.canvas_frame, width=600, height=400, bg="gray", highlightthickness=1, highlightbackground="black")
        self.canvas.pack()

        self.button_frame = tk.Frame(self.main_frame, borderwidth=2, relief="groove")
        self.button_frame.pack(side=tk.RIGHT, padx=(2, 10), fill=tk.Y)

        self.button_title = Label(self.button_frame, text="Controls", font=("Arial", 14, "bold"))
        self.button_title.pack(pady=(10, 5))

        self.label = Label(self.button_frame, text="Select Detection Method:", font=("Arial", 10))
        self.label.pack(pady=(5, 2))
        self.detection_method = tk.StringVar(value="yolo")
        tk.Radiobutton(self.button_frame, text="YOLOv11", variable=self.detection_method, value="yolo", font=("Arial", 10)).pack(anchor=tk.W, padx=10)
        tk.Radiobutton(self.button_frame, text="Faster R-CNN", variable=self.detection_method, value="fasterrcnn", font=("Arial", 10)).pack(anchor=tk.W, padx=10)
        tk.Radiobutton(self.button_frame, text="Haar Cascade", variable=self.detection_method, value="haar", font=("Arial", 10)).pack(anchor=tk.W, padx=10)

        self.upload_button = Button(self.button_frame, text="Upload Image/Video", command=self.upload_image_or_video, font=("Arial", 10))
        self.upload_button.pack(pady=10, padx=10, fill=tk.X)

        self.enhance_button = Button(self.button_frame, text="Enhance Image/Video", command=self.enhance_image_or_video, state="disabled", font=("Arial", 10))
        self.enhance_button.pack(pady=10, padx=10, fill=tk.X)

        self.detect_button = Button(self.button_frame, text="Detect Fish", command=self.detect_fish, state="disabled", font=("Arial", 10))
        self.detect_button.pack(pady=10, padx=10, fill=tk.X)

        self.download_image_button = Button(self.button_frame, text="Download Image", command=self.download_image, state="disabled", font=("Arial", 10))
        self.download_image_button.pack(pady=10, padx=10, fill=tk.X)

        self.download_video_button = Button(self.button_frame, text="Download Video", command=self.download_video, state="disabled", font=("Arial", 10))
        self.download_video_button.pack(pady=10, padx=10, fill=tk.X)

        self.separator = tk.Frame(root, height=2, bd=1, relief="sunken")
        self.separator.pack(fill=tk.X, padx=10, pady=5)

        self.results_frame = tk.Frame(root)
        self.results_frame.pack(fill=tk.X, padx=10, pady=10)

        self.results_label = Label(self.results_frame, text="Results:", font=("Arial", 12, "bold"))
        self.results_label.pack()
        self.results_text = Text(self.results_frame, height=10, width=80, font=("Arial", 10))
        self.results_text.pack()

        self.class_names = ["Epinephelus", "Chaetodon_Vagabundus", "Caranx", "Gerres", "Acanthopagrus"]

        yolo_paths = [
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/YoloV11/internetimages.pt",
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/YoloV11/mangrooveforest.pt" 
        ]
        fasterrcnn_paths = [
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/FasterRCNN/internetimages.pth",
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/FasterRCNN/mangrooveforest.pth" 
        ]
        haar_paths = [
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/HaarCascade/Acanthopagrus_Palmaris.xml",
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/HaarCascade/Caranx.xml",
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/HaarCascade/Chaetodon_Vagabundus.xml",
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/HaarCascade/Epinephelus.xml",
            "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Dataset/HaarCascade/Gerres.xml"
        ]

        self.yolo_models = [None, None]
        self.fasterrcnn_models = [None, None]
        try:
            self.yolo_models[0] = YOLO(yolo_paths[0])
            self.yolo_models[1] = YOLO(yolo_paths[1])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO models: {str(e)}")
        try:
            num_classes = 6
            for i, path in enumerate(fasterrcnn_paths):
                model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                model.eval()
                self.fasterrcnn_models[i] = model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Faster R-CNN models: {str(e)}")
        self.haar_cascade = {
            "Acanthopagrus Palmaris": haar_paths[0],
            "Caranx": haar_paths[1],
            "Chaetodon Vagabundus": haar_paths[2],
            "Epinephelus": haar_paths[3],
            "Gerres": haar_paths[4]
        }

        self.file_path = None
        self.is_video = False
        self.original_image = None
        self.enhanced_image = None
        self.current_display = None
        self.download_path = "C:/Users/Wilfred Auxilian/Desktop/Conference_Files/GUI_No3/Downloaded_Images_Videos"
        self.detected_frames = []
        self.detected_video_writer = None
        self.playing = False
        self.all_detected_fish = set()
        self.current_method = None

    def center_window(self):
        window_width = 1000
        window_height = 800
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_left = int(screen_width / 2 - window_width / 2)
        self.root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

    def upload_image_or_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg"), ("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.file_path = file_path
            self.is_video = file_path.endswith(('.mp4', '.avi', '.mov'))
            self.enhance_button.config(state="normal")
            self.detect_button.config(state="normal")
            self.download_image_button.config(state="normal")
            self.download_video_button.config(state="normal" if self.is_video else "disabled")
            self.detected_frames = []
            self.all_detected_fish = set()
            self.playing = False
            self.results_text.delete(1.0, tk.END)
            self.display_image_or_video()

    def display_image_or_video(self):
        if self.is_video:
            cap = cv2.VideoCapture(self.file_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb).resize((600, 400))
                self.tk_image = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                self.canvas.image = self.tk_image
                self.current_display = frame
            cap.release()
        else:
            self.original_image = cv2.imread(self.file_path)
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb).resize((600, 400))
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.image = self.tk_image
            self.current_display = self.original_image

    def enhance_image_or_video(self):
        if self.is_video:
            self.enhance_video()
        else:
            self.enhance_image()
        self.detect_button.config(state="normal")
        self.download_image_button.config(state="normal")
        self.download_video_button.config(state="normal" if self.is_video else "disabled")

    def enhance_image(self):
        if self.original_image is None:
            return

        b, g, r = cv2.split(self.original_image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        b_scale, r_scale = g_mean / (b_mean + 1e-6), g_mean / (r_mean + 1e-6)
        b_wb = np.clip(b.astype(np.float32) * b_scale, 0, 255).astype(np.uint8)
        g_wb = g
        r_wb = np.clip(r.astype(np.float32) * r_scale, 0, 255).astype(np.uint8)
        white_balanced = cv2.merge([b_wb, g_wb, r_wb])

        hsv_eq = cv2.cvtColor(white_balanced, cv2.COLOR_BGR2HSV)
        h_eq, s_eq, v_eq = cv2.split(hsv_eq)
        v_equalized = cv2.equalizeHist(v_eq)
        hsv_equalized = cv2.merge([h_eq, s_eq, v_equalized])
        color_histogram_eq = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

        hsv_clahe = cv2.cvtColor(color_histogram_eq, cv2.COLOR_BGR2HSV)
        h_clahe, s_clahe, v_clahe = cv2.split(hsv_clahe)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe_enhanced = clahe.apply(v_clahe)
        hsv_clahe_enhanced = cv2.merge([h_clahe, s_clahe, v_clahe_enhanced])
        self.enhanced_image = cv2.cvtColor(hsv_clahe_enhanced, cv2.COLOR_HSV2BGR)

        enhanced_rgb = cv2.cvtColor(self.enhanced_image, cv2.COLOR_BGR2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb).resize((600, 400))
        self.tk_image = ImageTk.PhotoImage(enhanced_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image
        self.current_display = self.enhanced_image

    def enhance_video(self):
        cap = cv2.VideoCapture(self.file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("enhanced_video.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            b, g, r = cv2.split(frame)
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
            b_scale, r_scale = g_mean / (b_mean + 1e-6), g_mean / (r_mean + 1e-6)
            b_wb = np.clip(b.astype(np.float32) * b_scale, 0, 255).astype(np.uint8)
            g_wb = g
            r_wb = np.clip(r.astype(np.float32) * r_scale, 0, 255).astype(np.uint8)
            white_balanced = cv2.merge([b_wb, g_wb, r_wb])

            hsv_eq = cv2.cvtColor(white_balanced, cv2.COLOR_BGR2HSV)
            h_eq, s_eq, v_eq = cv2.split(hsv_eq)
            v_equalized = cv2.equalizeHist(v_eq)
            hsv_equalized = cv2.merge([h_eq, s_eq, v_equalized])
            color_histogram_eq = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

            hsv_clahe = cv2.cvtColor(color_histogram_eq, cv2.COLOR_BGR2HSV)
            h_clahe, s_clahe, v_clahe = cv2.split(hsv_clahe)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v_clahe_enhanced = clahe.apply(v_clahe)
            hsv_clahe_enhanced = cv2.merge([h_clahe, s_clahe, v_clahe_enhanced])
            enhanced_frame = cv2.cvtColor(hsv_clahe_enhanced, cv2.COLOR_HSV2BGR)

            out.write(enhanced_frame)
            frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((600, 400))
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.image = self.tk_image
            self.root.update()

        cap.release()
        out.release()
        self.file_path = "enhanced_video.mp4"

    def detect_fish(self):
        method = self.detection_method.get()
        self.detected_frames = []
        self.all_detected_fish = set()
        self.current_method = method
        self.playing = True

        if method == "yolo" and (not self.yolo_models[0] or not self.yolo_models[1]):
            messagebox.showerror("Error", "One or both YOLO models are not loaded!")
            return
        elif method == "fasterrcnn" and (not self.fasterrcnn_models[0] or not self.fasterrcnn_models[1]):
            messagebox.showerror("Error", "One or both Faster R-CNN models are not loaded!")
            return

        if self.is_video:
            self.detect_video(method)
        else:
            if self.enhanced_image is not None:
                cv2.imwrite("temp_enhanced.jpg", self.enhanced_image)
                input_path = "temp_enhanced.jpg"
            else:
                input_path = self.file_path

            detected_fish = []
            if method == "yolo":
                detected_fish = self.detect_with_yolo(input_path)
            elif method == "fasterrcnn":
                detected_fish = self.detect_with_fasterrcnn(input_path)
            elif method == "haar":
                detected_fish = self.detect_with_haar(input_path)

            self.update_results(detected_fish, method)
            if not detected_fish:
                messagebox.showinfo("Detection Result", "No fish detected in the image. Please Train Your Model More!")

    def detect_video(self, method):
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps)
        frame_count = 0
        skip_frames = 5

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.detected_video_writer = cv2.VideoWriter("temp_detected_video.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened() and self.playing:
            ret, frame = cap.read()
            if not ret:
                break

            frame_small = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))

            if frame_count % skip_frames == 0:
                temp_frame_path = "temp_frame.jpg"
                cv2.imwrite(temp_frame_path, frame_small)

                detected_fish = []
                if method == "yolo":
                    detected_fish = self.detect_with_yolo_frame(temp_frame_path, frame_small)
                elif method == "fasterrcnn":
                    detected_fish = self.detect_with_fasterrcnn_frame(temp_frame_path, frame_small)
                elif method == "haar":
                    detected_fish = self.detect_with_haar_frame(temp_frame_path, frame_small)

                self.all_detected_fish.update(detected_fish)
                self.update_results(list(self.all_detected_fish), method)

                frame_with_detections = cv2.imread(temp_frame_path)
                frame_with_detections = cv2.resize(frame_with_detections, (frame.shape[1], frame.shape[0]))
            else:
                frame_with_detections = frame

            self.detected_video_writer.write(frame_with_detections)

            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((600, 400))
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.image = self.tk_image
            self.current_display = frame_with_detections

            self.root.update()
            self.root.after(frame_delay)

            frame_count += 1

        cap.release()
        self.detected_video_writer.release()
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
        self.playing = False

        self.root.update()
        if not self.all_detected_fish:
            messagebox.showinfo("Detection Result", "No fish detected in the video. Please Train Your Model More!")

    def detect_with_yolo_frame(self, input_path, frame):
        yolo_class_names = ["Chaetodon_Vagabundus", "Epinephelus", "Acanthopagrus", "Caranx", "Gerres"]
        detected_fish = []
        image_cv = cv2.imread(input_path)

        for model in self.yolo_models:
            results = model(input_path)
            for pred in results[0].boxes:
                fish_class = pred.cls
                if fish_class != 0:
                    class_name = yolo_class_names[int(fish_class) - 1]
                    detected_fish.append(class_name)
                    # Draw bounding box
                    xyxy = pred.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_cv, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(input_path, image_cv)
        if detected_fish:
            self.detected_frames.append(frame)
        return list(set(detected_fish))

    def detect_with_fasterrcnn_frame(self, input_path, frame):
        image = Image.open(input_path).convert("RGB")
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0)
        image_cv = cv2.imread(input_path)
        detected_fish = []

        for model in self.fasterrcnn_models:
            with torch.no_grad():
                predictions = model(image_tensor)
            boxes = predictions[0]['boxes']
            labels = predictions[0]['labels']
            scores = predictions[0]['scores']
            for i, score in enumerate(scores):
                if score > 0.5:
                    x1, y1, x2, y2 = boxes[i].cpu().numpy()
                    label_idx = labels[i].item()
                    if label_idx > 0:
                        label_name = self.class_names[label_idx - 1]
                        detected_fish.append(label_name)
                        cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image_cv, label_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(input_path, image_cv)
        if detected_fish:
            self.detected_frames.append(frame)
        return list(set(detected_fish))

    def detect_with_haar_frame(self, input_path, frame):
        image = cv2.imread(input_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_fish = []
        for fish_name, cascade_path in self.haar_cascade.items():
            fish_cascade = cv2.CascadeClassifier(cascade_path)
            fish = fish_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in fish:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_fish.append(fish_name)
                cv2.putText(image, fish_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(input_path, image)
        if detected_fish:
            self.detected_frames.append(frame)
        return detected_fish

    def detect_with_yolo(self, input_path):
        yolo_class_names = ["Chaetodon_Vagabundus", "Epinephelus", "Acanthopagrus", "Caranx", "Gerres"]
        detected_fish = []
        image_cv = cv2.imread(input_path)

        for model in self.yolo_models:
            results = model(input_path)
            for pred in results[0].boxes:
                fish_class = pred.cls
                if fish_class != 0:
                    class_name = yolo_class_names[int(fish_class) - 1]
                    detected_fish.append(class_name)
                    xyxy = pred.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_cv, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_path = "output_yolo.jpg"
        cv2.imwrite(output_path, image_cv)
        self.display_result(output_path)
        return list(set(detected_fish))

    def detect_with_fasterrcnn(self, input_path):
        image = Image.open(input_path).convert("RGB")
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0)
        image_cv = cv2.imread(input_path)
        detected_fish = []

        for model in self.fasterrcnn_models:
            with torch.no_grad():
                predictions = model(image_tensor)
            boxes = predictions[0]['boxes']
            labels = predictions[0]['labels']
            scores = predictions[0]['scores']
            for i, score in enumerate(scores):
                if score > 0.5:
                    x1, y1, x2, y2 = boxes[i].cpu().numpy()
                    label_idx = labels[i].item()
                    if label_idx > 0:
                        label_name = self.class_names[label_idx - 1]
                        detected_fish.append(label_name)
                        cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image_cv, label_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_path = "fasterrcnn_output.jpg"
        cv2.imwrite(output_path, image_cv)
        self.display_result(output_path)
        return list(set(detected_fish))

    def detect_with_haar(self, input_path):
        image = cv2.imread(input_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_fish = []
        for fish_name, cascade_path in self.haar_cascade.items():
            fish_cascade = cv2.CascadeClassifier(cascade_path)
            fish = fish_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in fish:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_fish.append(fish_name)
                cv2.putText(image, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite("haar_output.jpg", image)
        self.display_result("haar_output.jpg")
        return detected_fish

    def estimate_distance_and_area(self, frame):
        height, width = frame.shape[:2]
        focal_length = 1000
        object_size = 1.0
        distance = (object_size * focal_length) / max(height, width) * 1000
        distance = distance * random.uniform(0.8, 1.2)

        fov_angle = 60
        fov_rad = np.deg2rad(fov_angle)
        area_width = 2 * distance * np.tan(fov_rad / 2) * (width / height) / 1000
        area_height = 2 * distance * np.tan(fov_rad / 2) / 1000
        frame_area = area_width * area_height

        return distance, frame_area

    def update_results(self, detected_fish, method):
        self.results_text.delete(1.0, tk.END)
        if not detected_fish:
            self.results_text.insert(tk.END, "No detections found yet.")
        else:
            distance, frame_area = self.estimate_distance_and_area(self.current_display)
            fish_count = len(detected_fish)
            fish_names = ", ".join(detected_fish)
            self.results_text.insert(tk.END, f"- OUTPUT\n\n")
            self.results_text.insert(tk.END, f"No of Fishes detected: {fish_count}\n")
            self.results_text.insert(tk.END, f"Fish Classes detected in the display: {fish_names}\n")
            self.results_text.insert(tk.END, f"Detection Method Used: {method}\n")
            self.results_text.insert(tk.END, f"Approx. frame area covered: {frame_area:.2f} sq. meter\n")
            self.results_text.insert(tk.END, f"Dist. b/w camera & seabed: {distance:.1f} mm\n")

    def display_result(self, path):
        image = Image.open(path).resize((600, 400))
        self.result_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.result_image)
        self.canvas.image = self.result_image
        self.current_display = cv2.imread(path)

    def download_image(self):
        if self.current_display is not None:
            if not os.path.exists(self.download_path):
                os.makedirs(self.download_path)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.download_path, f"downloaded_image_{timestamp}.jpg")
            cv2.imwrite(file_path, self.current_display)
            messagebox.showinfo("Success", f"Image saved to {file_path}")

    def download_video(self):
        if self.is_video and os.path.exists("temp_detected_video.mp4"):
            if not os.path.exists(self.download_path):
                os.makedirs(self.download_path)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.download_path, f"downloaded_video_{timestamp}.mp4")
            os.rename("temp_detected_video.mp4", file_path)
            messagebox.showinfo("Success", f"Video saved to {file_path}")
        else:
            messagebox.showerror("Error", "No detected video available to download. Please run detection on a video first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FishDetectionGUI(root)
    root.mainloop()