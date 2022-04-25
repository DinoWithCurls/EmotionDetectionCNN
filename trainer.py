import cv2
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)
#Preprocess all training images
trainer_generator = train_data.flow_from_directory(
    'imgDataset/train',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)
test_generator = test_data.flow_from_directory(
    'imgDataset/test',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)
#Create Model Structure
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#check dropout and dense, activation methods
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

model_train = model.fit_generator(
    trainer_generator,
    steps_per_epoch=31805//64,
    epochs=100,
    validation_data=test_generator,
    validation_steps=7178//64)
#steps_per_epoch = no. of files in the training dataset // batch size
#validation_steps = no. of files in the test dataset // batch size
jsonmodel = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(jsonmodel)

model.save_weights('model.h5')