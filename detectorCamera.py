import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

model.load_weights('model.h5')
print("Loaded model")

cam_Cap = cv2.VideoCapture(0)

while True:
    ret, frame = cam_Cap.read()
    frame = cv2.resize(frame, (800, 600))
    if not ret:
        break
    #create Haar Cascade object
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    #convert the frame taken from webcam into grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #get the face from frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x,y, width, height) in faces:
        #preprocess the faces available from the frame
        cv2.rectangle(frame, (x,y-50), (x+width, y+height+10), (0,255,0), 4)
        roi_gray_frame = gray_frame[y:y+height, x:x+width]
        cropped_image = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)
        #get predictions
        prediction = model.predict(cropped_image)
        maxidx = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxidx], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_Cap.release()
cv2.destroyAllWindows()


