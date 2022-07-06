# Emotion Detection in Humans using CNN

## Technologies Used
1. Python
2. OpenCV
3. Tensorflow-GPU (depending on GPU availability)
4. NVidia CUDA Computing Toolkit (depending on GPU availability)

## Dataset Used
FER_Modified : https://www.kaggle.com/srinivasbece/fer-modified

## Group Members
1. Syed Qausain Huda
2. Swadha Kumar
3. Saunak Das
4. Aditya Raj Singh

## Description
This is an implementation of Convolutional Neural Network architecture for detecting emotions in human faces, provided either as a static image or as a video from webcam. The model is trained beforeuse, and then it is provided the images for predicting. The prediction value is passed back, which decides on the emotion being displayed by the person. Our model has one input layer, 4 hidden layers, and two fully-connected neural networks(FCNNs). Through this, we were able to achieve around 97% testing accuracy and around 78% validation accuracy. We use Tensorflow and Keras functions and models for our purpose.

## Testing and Validation Graphs
### For training at 20 epochs
![20epochs](https://user-images.githubusercontent.com/42311383/174627233-b124edd4-e48a-4e53-9dc7-ea19052d2df1.png)

### For training at 50 epochs
![50epochs](https://user-images.githubusercontent.com/42311383/174627812-44108dd9-c8b0-4ff9-a6b9-06bc1a3b3ac7.png)

### For training at 100 epochs
![100epochs](https://user-images.githubusercontent.com/42311383/174628042-697c7266-11a7-4ff6-9c38-410d76b2a630.png)


## Results we got

![res1](https://user-images.githubusercontent.com/42311383/174628687-8bdecfee-877c-4e57-b08c-f44b3105f777.jpg)  ![res2](https://user-images.githubusercontent.com/42311383/174628697-87850430-3c51-4b23-8a5e-c88066c16663.jpg)

## Future Scope
1. Our training was done only on static images only having the faces. We can have our model trained with videos.
2. Our model currently is not great with side profiles. We can either train it using side profiles as well, or use better libraries for the purpose. 
3. Our model does not work properly in cases of bad lighting. We need to work on that.
4. We can implement this model for other forms of media as well, such as audio, video, text.
