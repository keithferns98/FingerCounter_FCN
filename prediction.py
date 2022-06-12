import cv2 as cv
import tensorflow as tf
import numpy as np
import operator
model2 = tf.keras.models.load_model('fin.h5')
print('model loaded')

cap=cv.VideoCapture(0)

categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

while True:
    _,frame=cap.read()
    frame=cv.flip(frame,1)

    #Co-ordinates of the roi
    x1= int(0.5*frame.shape[1])
    y1=30
    x2=frame.shape[1]-10
    y2=int(0.5*frame.shape[1])
    cv.rectangle(frame, (x1-1, y1-1), (x2-1, y2-1), (255,0,0) ,1)
    print(x1,y1,x2,y2)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi=cv.resize(roi,(64,64))
    roi=cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
    _, test_image=cv.threshold(roi,120,255,cv.THRESH_BINARY)
    # cv.imshow("test",test_image)
    
    result=model2.predict(test_image.reshape(1,64,64,1))
    prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5]}
    prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)
    print(prediction)
    cv.putText(frame, prediction[0][0], (10, 120), cv.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
    print(prediction[0][0])
    cv.imshow("Frame",frame)
    interrupt=cv.waitKey(10)
    if interrupt & 0xFF ==27:
        break

cap.release()
cv.destroyAllWindows()
    

