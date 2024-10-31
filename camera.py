import cv2

index = 0
arr = []
while index < 2:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1
print(f"Available camera indexes: {arr}")
