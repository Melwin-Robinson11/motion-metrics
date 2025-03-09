import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Video path
video_path = r"D:\Projects\RealTimeTrafficAnalysis\CarsOnTheHighway.mp4"
cap = cv2.VideoCapture(video_path)

# Speed calculation variables
detection_time = {}
speeds = []

# Speed constants
fps = cap.get(cv2.CAP_PROP_FPS)
meters_per_pixel = 0.02  # Adjust this based on your video scale

# ROI Definition
def define_roi(frame):
    height, width = frame.shape[:2]
    return [(0, int(height * 0.5)), (width, int(height * 0.8))]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    detections = results[0].boxes.data  # Detected objects

    roi = define_roi(frame)
    cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)  # Green ROI box

    for detection in detections:
        x, y, w, h, conf, cls = detection
        class_id = int(cls)

        # Detect cars or trucks
        if class_id == 2 or class_id == 7:
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 0), 2)

            # Display Class ID above the vehicle
            cv2.putText(frame, f"ID: {class_id}", (int(x), int(y) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            vehicle_id = f"{int(x)}-{int(y)}"
            if vehicle_id not in detection_time:
                detection_time[vehicle_id] = time.time()
            else:
                elapsed_time = time.time() - detection_time[vehicle_id]

                # 🔹 Prevent ZeroDivisionError by ensuring elapsed_time is > 0
                if elapsed_time > 0.1:
                    distance_meters = (roi[1][1] - int(y)) * meters_per_pixel
                    speed_kph = (distance_meters / elapsed_time) * 3.6
                    speeds.append(speed_kph)

                    cv2.putText(frame, f"{speed_kph:.2f} km/h", (int(x), int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Vehicle Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Display average speed
if speeds:
    avg_speed = sum(speeds) / len(speeds)
    print(f"Average Speed: {avg_speed:.2f} km/h")
else:
    print("No vehicles detected.")
