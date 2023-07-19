import cv2

cap = cv2.VideoCapture(r'rtsp://192.168.12.100:554/stream1')
print(cap)
ret,frame = cap.read()
while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        cv2.imshow('img',frame)
        cv2.waitKey(1)

    print('we are in')
    print(frame.shape)
    if not ret:
        print('error')
        break
    