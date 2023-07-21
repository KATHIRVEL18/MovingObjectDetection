import cv2
import time
import imutils


Camera = cv2.VideoCapture(0)

time.sleep(1)

firstFrame = None
area = 500
ObjectCount = 0
while True:
    _,Image = Camera.read()
    text = "Normal"
    Image = imutils.resize(Image, width=500)
    GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    GaussianImage = cv2.GaussianBlur(GrayImage, (21,21), 0)
    if firstFrame is None:
        firstFrame = GaussianImage
        continue

    ImageDif = cv2.absdiff(firstFrame, GaussianImage)
    
    ThresholdImage = cv2.threshold(ImageDif, 25, 255, cv2.THRESH_BINARY)[1]
    
    ThresholdImage = cv2.dilate(ThresholdImage, None, iterations=2)
    
    contours = cv2.findContours(ThresholdImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = imutils.grab_contours(contours)
    
    for c in contours :
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(Image, (x,y), (x + w, y + h), (0, 0, 255), 2)
        text = "Moving Object Is Detected"
        ObjectCount += 1
        
    print(text + "with count no :" + str(ObjectCount))
    
    cv2.putText(Image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.putText(Image, str(ObjectCount), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    cv2.imshow("CameraFeed",Image)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break;
    

Camera.release()
cv2.destroyAllWindows()
