<img width="644" alt="Screen Shot 2022-04-04 at 11 39 07 AM" src="https://user-images.githubusercontent.com/48040161/161465092-96cd7918-8508-4e33-bdf8-06b385ce9851.png">

# Brasa Hacks 2022

## AI Implementation

The in-line comments are present in the code.

Before using the code, make sure to install all the necessary dependencies.
```
pip3 install -r requirements.txt
```

In order to train the AI, you must execute the following command:
```
python3 train.py
```

The training algorithm uses a dataset of 1000 labeled images stored in the IBM Annotations platform. After requesting the images, and labels, the algorithm uses TensorFlow and Keras to deliver a trained model for the image classification.

## Hosting the model

After the model was trained, we upload it to an AWS virtual machine, where we create an access point for receiving the images, processing them, and using the model to infer the risk level of the water, returning the prediction to the rest of our environment.
