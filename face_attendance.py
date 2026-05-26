import cv2
import face_recognition
import os
import csv
from datetime import datetime

KNOWN_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

known_encodings = []
known_names = []

# Load known faces
for file in os.listdir(KNOWN_DIR):
    if file.lower().endswith((".jpeg", ".avif")):
        path = os.path.join(KNOWN_DIR, file)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])

print("Known faces:", known_names)

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "r") as f:
        records = f.readlines()

    for line in records[1:]:
        entry_name, entry_date, _ = line.strip().split(",")
        if entry_name == name and entry_date == today:
            return  # already marked today

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now_time])
        print(f"Attendance marked for {name}")

# Open phone camera (change index if needed)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)


if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )

        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]
            mark_attendance(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame, name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
