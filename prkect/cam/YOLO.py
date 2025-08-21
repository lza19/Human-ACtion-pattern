import cv2
from ultralytics import YOLO

#model = YOLO('yolov8n.pt')
model = YOLO('yolov9c.pt')

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

    # นำเฟรมไปให้โมเดลตรวจจับคน
    # show=False เพราะเราจะแสดงผลด้วยตัวเองทีหลัง
    results = model(frame, stream=True,classes=[0])

    # วาดกรอบและแสดงผลลัพธ์
    for result in results:
        # วาดกรอบสี่เหลี่ยมรอบคนที่ตรวจจับได้
        annotated_frame = result.plot(labels=True, conf=True)

        # แสดงเฟรมที่ถูกวาดกรอบแล้วในหน้าต่างชื่อ 'YOLO Live'
        cv2.imshow("YOLO Live", annotated_frame)
    
    # รอการกดปุ่ม 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างที่แสดงผล
cap.release()
cv2.destroyAllWindows()