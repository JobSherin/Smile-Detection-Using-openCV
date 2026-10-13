import cv2
import os

capture = cv2.VideoCapture(0)

facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smiledetector = cv2.CascadeClassifier('haarcascade_smile.xml')

savepath = 'captured_face'
os.makedirs(savepath,exist_ok=True)

imagecount = 0

smile_detected = False

while(1):
    ret,img = capture.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetector.detectMultiScale(gray,1.5,5)

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

        face_gray = gray[y:y+h, x:x+w]

        mouth_area = face_gray[int(h/2):h,:]

        smiles = smiledetector.detectMultiScale(mouth_area,1.7,30)

        if len(smiles)>0:
         cv2.putText(img,"smile Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)


         if not smile_detected:


          imagecount +=1

          filename = os.path.join(savepath,f"smile_{imagecount}.jpg")
          result =cv2.imwrite(filename,img)

          if result:
             print("Image saved",filename)
          else:
             print("Image not saved")

          smile_detected= True

        else:
           smile_detected = False

    cv2.imshow('My Frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()