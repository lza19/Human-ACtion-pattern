import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO('yolov8n.pt')

# เปิดกล้องเว็บแคม (0 คือกล้องตัวแรก)
cap = cv2.VideoCapture(0)

# เช็คว่ากล้องเปิดใช้งานได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้ โปรดตรวจสอบการเชื่อมต่อกล้อง")
    exit()

while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านเฟรมจากกล้องได้")
        break

    # นำเฟรมไปให้โมเดลตรวจจับและติดตามคน
    # โดยใช้พารามิเตอร์ tracker='bytetrack.yaml' หรือ 'botsort.yaml'
    # botsort จะมีความแม่นยำสูง แต่ bytetrack จะทำงานได้เร็วกว่า
    results = model.track(frame, classes=[0], persist=True, tracker="bytetrack.yaml")

    # วาดกรอบและแสดงผลลัพธ์
    for result in results:
        # result.plot() จะวาดกรอบสี่เหลี่ยมพร้อม ID ให้โดยอัตโนมัติ
        annotated_frame = result.plot()
        cv2.imshow("YOLO Live", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()