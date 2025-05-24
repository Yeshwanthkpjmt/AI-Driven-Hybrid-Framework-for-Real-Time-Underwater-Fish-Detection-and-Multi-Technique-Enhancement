![image](https://github.com/user-attachments/assets/2afe6782-1c62-4703-8392-e27e5462e13a)


This study introduces a hybrid framework, powered by AI, for the real-time detection and identification of fish in underwater, with an added capability of multi-technique image enhancement. The hybrid framework contains three different object detection ML algorithms YOLOv11, Faster R-CNN and Haar-Cascade to classify Underwater fish with the abundance and to explore the difficulties of collecting data in underwater imaging. For the detection process to apply effectively in real-timescenarios, the study introduces a new image enhancement pipeline that will include white balancing, histogram equalization and a special form of histogram equalization called Contrast Limited Adaptive Histogram Equalization (CLAHE). The hybrid framework is written in Python utilizing Tkinter for a graphical user interface (GUI) and OpenCV for the image processing. The user-friendly GUI allows user to input both images and videos,which enables the hybrid framework to utilize in real-time, while the user can visualize the fish right away as well as download the results after completion to save for future analysis. The results obtained from the hybrid framework support the observation that YOLOv11 process the fish faster - which is a primary concern in real-time applications, while Faster R-CNN was the most accurate method in detecting and classifying fish species in more complex scenes, and the Haar Cascade offered a fast and lightweight solution. Overall, the hybrid detection framework can provide a better detection experience for fish while allowing for increased detection and classification abilities, which may provide practical solutions for researching marine biology and effective management of aquaculture.


Main Topics in This Project

Object Detection Models:-
YOLOv11 – Fastest model, suitable for real-time applications.
Faster R-CNN – Most accurate model, suitable for complex underwater scenes.
Haar Cascade – Lightweight and fast, suitable for basic or resource-constrained environments.


Image Enhancement Techniques:-
White Balancing – Corrects color distortion in underwater images.
Histogram Equalization – Improves brightness and contrast.
CLAHE (Contrast Limited Adaptive Histogram Equalization) – Enhances local contrast in images.


Dataset Preparation:-
Used tools like LabelImg and Roboflow for annotation.
Fish species used: Acanthopagrus palmaris, Caranx, Chaetodon Vagabundus, Epinephelus, Gerres.


Graphical User Interface (GUI) - Built using Tkinter and Allows:-
Input of images/videos.
Model selection.
Real-time result visualization.
Export of annotated results.


The comparison of 3 algorithm suing our model:-

| **Metric**      | **YOLOv11**      | **Faster R-CNN** | **Haar Cascade** |
| --------------- | ---------------- | ---------------- | ---------------- |
| **Precision**   | 60.00% – 60.93%  | 55.00% – 58.00%  | 10.00% – 12.00%  |
| **Recall**      | 98.45% – 100.00% | 90.00% – 92.00%  | 7.75% – 11.63%   |
| **Accuracy**    | 60.00% – 61.00%  | 58.00% – 60.00%  | 14.00% – 19.00%  |
| **F1-Score**    | 0.75             | 0.71 – 0.73      | 0.14 – 0.19      |
| **FPS (Image)** | 2.79 – 3.27      | < 0.3            | 0.15 – 0.24      |
| **FPS (Video)** | 5.63 – 6.19      | < 0.3            | 0.15 – 0.24      |


📜 LICENSE:-

⚠️ Strict Usage Policy

🔒 This project and all its contents — including **code**, **images**, **data**, and **results** — are **fully protected**.

You **MAY NOT**:
- Reuse, reproduce, modify, or redistribute any part of this project.
- Use it for **personal**, **academic**, or **commercial** gain **without written consent** from the author.

💼 Unauthorized commercial use will lead to **legal action**.

📧 Contact for Permissions
To request permission for use, contact:  
**👉 yeshwanthkpjmt@gmail.com**

---

