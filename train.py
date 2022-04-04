# Import required libraries
import os
import uuid
import shutil
import json
from botocore.client import Config
import ibm_boto3
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
from PIL import ImageDraw 

# Connect to IBM annotations
credentials = {
  "bucket": "green-elephant",
  "access_key_id": "49f7434ab6874ce2a6928f3bb8bafe80",
  "secret_access_key": "a67f7eae15b33399835f0c01340ed041baa9b45c4f77b005",
  "endpoint_url": "https://s3.us.cloud-object-storage.appdomain.cloud"
}


# Read The Data

def download_file_cos(credentials, local_file_name, key): 
    cos = ibm_boto3.client(
        service_name='s3',
        aws_access_key_id=credentials['access_key_id'],
        aws_secret_access_key=credentials['secret_access_key'],
        endpoint_url=credentials['endpoint_url'])
    try:
        res=cos.download_file(Bucket=credentials['bucket'], Key=key, Filename=local_file_name)
    except Exception as e:
        print(Exception, e)
    else:
        print('File Downloaded')

def get_annotations(credentials): 
    cos = ibm_boto3.client(
        service_name='s3',
        aws_access_key_id=credentials['access_key_id'],
        aws_secret_access_key=credentials['secret_access_key'],
        endpoint_url=credentials['endpoint_url'])
    try:
        return json.loads(cos.get_object(Bucket=credentials['bucket'], Key='_annotations.json')['Body'].read())
    except Exception as e:
        print(Exception, e)

base_path = 'data'
if os.path.exists(base_path) and os.path.isdir(base_path):
    shutil.rmtree(base_path)
os.makedirs(base_path, exist_ok=True)

annotations = get_annotations(credentials)

for i, image in enumerate(annotations['annotations'].keys()):
    label = annotations['annotations'][image][0]['label']
    os.makedirs(os.path.join(base_path, label), exist_ok=True)
    _, extension = os.path.splitext(image)
    local_path = os.path.join(base_path, label, str(uuid.uuid4()) + extension)
    download_file_cos(credentials, local_path, image)

# Layers init
base_model=tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False)
x=base_model.output
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(512,activation='relu')(x) 
x=tf.keras.layers.Dense(256,activation='relu')(x) 
preds=tf.keras.layers.Dense(3,activation='softmax')(x) 


for layer in base_model.layers:
    layer.trainable=False

# Prepare the training dataset and model
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('data',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=15,
                                                 class_mode='categorical',
                                                 shuffle=True)


model = tf.keras.Sequential([
  hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4", 
                 output_shape=[1280],
                 trainable=False),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.build([None, 224, 224, 3])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])


# Train the model

steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)

log_file = model.fit(
    train_generator, 
    epochs=50,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=train_generator,
    validation_steps=val_steps_per_epoch).history

# Plot figure of Training Loss and Accuracy




model.save('model.h5')
print(log_file)
plt.plot(log_file['acc'], '-bo', label="train_accuracy")
plt.plot(log_file['loss'], '-r*', label="train_loss")
plt.title('Training Loss and Accuracy')
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epoch #')
plt.legend(loc='center right')
plt.show()

# Mapping labels 
label_map = (train_generator.class_indices)

label_map

# Testing

def prediction(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    
    for pred, value in label_map.items():    
        if value == np.argmax(preds):
            print('Predicted class is:', pred)
            print('With a confidence score of: ', np.max(preds))
    
    return (pred, np.max(preds))

path = "flood.png"
pred, preds = prediction(path, model)
image = Image.open(path)
draw = ImageDraw.Draw(image)
image

