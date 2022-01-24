import cv2

plateCascade=cv2.CascadeClassifier('./indian_license_plate.xml')
minArea=500

vid=cv2.VideoCapture(0)

while True:
    ret, frame=vid.read()
    imgGray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates=plateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in plates:
        area=w*h
        if area>minArea:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=2)

    cv2.imshow('Automatic License Plate Detection', frame)
    cv2.waitKey(1)
