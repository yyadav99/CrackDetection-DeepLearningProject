# CrackDetection
Basic Deep Neural Network Model to identify image has crack or not.

## Content
  - Overview
  - Prerequisites
  - Flow Diagram
  - Dataset
  - Steps to use pre-trained model
  - TODO
  
### Overview
Crack segmentation is an important task in structure investigation problems. For example, in the bridge investigation project, a drone is controlled to fly around bridges to take pictures of different bridge surfaces. The pictures will be then processed by a computer to detect potential regions on the bridge surface that might be damaged. The more accurate the model is, the less human effort we need to process these images. Otherwise, the operators will have to check every single image, which is boring and error-prone. Challenges in this task are first, the model is sensitive to noise and other objects such as moss on crack, title lines, etc, and secondly, get high-resolution images. In this project, I  trained our model with 20000 crack images and 20000 no-crack images datasets available on the internet. The result shows that the model could be able to distinguish crack from a tree and other different cracks on any surfaces.
### Prerequisites 
1. Python Libraries:
   - Numpy
   - Pandas
   - MatplotLib/Seaborn
   - Sklearn
   - OpenCV
     - How to read an image
     - How to display an image
     - How to resize an image
   - Keras
2. What is CNN and how does it work.
3. TQDM for better visualization of loops.


### Flow Diagram 
This is a basic flow diagram of how the code is being implemented.
1. Original Image is read.
2. Image is converted into grayscale.
3. Image is reshaped.
4. The reshaped image is fed to the model for prediction.
5. The model predicts whether the image has crack or not.
 
 
![flowdigram1](https://user-images.githubusercontent.com/51474690/84495687-340ea380-acc9-11ea-83f4-320646ab9bfb.jpg)


### Dataset
You can find the dataset from here :-
1. [Link 1](https://drive.google.com/file/d/1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP/view?usp=sharing)
2. [Link 2](https://drive.google.com/file/d/1kC60RGO3rcScVk7HY-s7tTMJeMbADfh1/view?usp=sharing)
3. [Link 3](https://drive.google.com/drive/folders/1XZ4VBVs1YQ0gZgLgOfMuMgsI-w20yxfZ?usp=sharing)

>Personally I have used Link 3 to train this model.

### Steps to use pre-trained model
1. Download the pre-trained model from [here](https://drive.google.com/file/d/1LByaJXD6i4X3uACwUzKidNi2QJ69Uy1S/view?usp=sharing)
2. You can predict only 1 image at a time for more images you can run a for loop. You just have to define the following function.
```
def predict_image(a):
    import cv2
    import tensorflow as tf
    model = tf.keras.models.load_model('CrackDetection.h5')
    img = cv2.imread(a ,0) 
    img = cv2.resize(img,(100,75)).reshape(-1, 100,75, 1)
    classes = model.predict_classes(img)
    if classes == 0:
        return  print("NON-CRACK IMAGE")
    else:
        return print("CRACK IMAGE")
```
3. Then call the function with 1 parameter containing the path of the variable.
```
path="file_name.ext"
predicted_value =predict_image(path)
```

## TODO

- [ ] Implement the model in the android application.
- [ ] Find the length of the crack.

**If you encounter any issue while using code or model, feel free to contact me yadavyogesh9999@gamil.com.**
