import cv2
import numpy as np
import os

#=================================================================================================================================================================================================================
# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
   
    
#=================================================================================================================================================================================================================
# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

#=================================================================================================================================================================================================================
while True:
    _, frame = cap.read()
    kernel = np.ones((1,1),np.uint8)
    
    #---------------------------------------------------------------------------------------------------
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    #---------------------------------------------------------------------------------------------------
    # Getting count of existing images
    count = {'Victory': len(os.listdir(directory+"/0")),
             'Thumb': len(os.listdir(directory+"/1")),
             'Fist': len(os.listdir(directory+"/2")),
             'Palm': len(os.listdir(directory+"/3")),}

    #---------------------------------------------------------------------------------------------------
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)
    cv2.putText(frame, "Victory : "+str(count['Victory']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)
    cv2.putText(frame, "Thumb : "+str(count['Thumb']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)
    cv2.putText(frame, "Fist : "+str(count['Fist']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)
    cv2.putText(frame, "Palm : "+str(count['Palm']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)
   
    #---------------------------------------------------------------------------------------------------
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])

    #---------------------------------------------------------------------------------------------------
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
    cv2.imshow("Frame", frame)

    #---------------------------------------------------------------------------------------------------
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['Victory'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['Thumb'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['Fist'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['Palm'])+'.jpg', roi)
   
    
cap.release()
cv2.destroyAllWindows()
