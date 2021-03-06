from flask import Flask, request
import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub
import os
from io import BytesIO
from PIL import Image


model = tf.keras.models.load_model('model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
print("Model Loaded")

app = Flask(__name__)

@app.route("/model", methods=['POST'])
def serve_model():
  try:

    file = request.files['image'].read()

    temp_path = "temp.png"
    
    image = Image.open(BytesIO(file))
    image.save(temp_path)

    img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    preds = model.predict(x)

    label_map = {'dangerously deep': 0,
    'feet-dont-touch deep': 1,
    'knee deep': 2,
    'waist deep': 3}
    
    for pred, value in label_map.items():    
        if value == np.argmax(preds):
            print('Predicted class is:', pred)
            print('With a confidence score of: ', np.max(preds))
            return str(pred.replace(" ", "_").replace("-", "_")) +" "+ str(np.max(preds))

    
  except Exception as e:
    print(e)
    return "Error"

 
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)