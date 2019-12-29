import cv2, time, pandas
from datetime import datetime

base_frame = None
video = cv2.VideoCapture(0)

status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if base_frame is None:
        base_frame = gray
        continue

    delta_frame = cv2.absdiff(base_frame, gray)
    threshold_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
    (cnts,_) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 5000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now()) 
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    status_list.append(status)
    cv2.imshow("Threshold",threshold_frame)
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Original Frame", frame)

    key = cv2.waitKey(1)

    if key==ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

for i in range(0, len(times), 2):
    df = df.append({"Start" : times[i], "End" : times[i+1]}, ignore_index = True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows