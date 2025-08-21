import torch
print(torch.__version__)
# ตรวจสอบว่า PyTorch สามารถเข้าถึง GPU ได้หรือไม่
if torch.cuda.is_available():
    print("PyTorch สามารถมองเห็น GPU ได้!")
    print(f"GPU ที่ใช้: {torch.cuda.get_device_name(0)}")
    print(f"จำนวน GPU ที่พบ: {torch.cuda.device_count()}")
else:
    print("PyTorch ไม่สามารถมองเห็น GPU ได้ กรุณาตรวจสอบการติดตั้ง CUDA")