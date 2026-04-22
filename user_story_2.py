import cv2
print("OpenCV version:", cv2.__version__)

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("❌ Haar cascade NOT loaded")
    exit()
else:
    print("✅ Haar cascade loaded")

# Try camera index 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera 0 failed, trying camera 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ No camera accessible")  
    exit()

print("✅ Camera opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    print("Faces detected:", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Forehead ROI
        fx1 = x + int(0.2 * w)
        fx2 = x + int(0.8 * w)
        fy1 = y
        fy2 = y + int(0.3 * h)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

    cv2.imshow("User Story 2 - DEBUG", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



