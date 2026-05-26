import cv2
from fer import FER

# Initialize emotion detector
emotion_detector = FER(mtcnn=False)

# Load Haar face detector
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

# ⚠️ USE YOUR WORKING CAMERA INDEX
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)  # change if needed

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Emotion detection started (FER)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        emotions = emotion_detector.detect_emotions(face_img)

        if emotions:
            emotion, score = max(
                emotions[0]["emotions"].items(),
                key=lambda x: x[1]
            )
            label = f"{emotion} ({score:.2f})"
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection (FER)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
