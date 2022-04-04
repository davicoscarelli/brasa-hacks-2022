from flask import Flask, request
import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub
import os
from io import BytesIO
from PIL import Image


model = tf.keras.models.load_model('model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
print("loaded model")

app = Flask(__name__)
@app.route("/model", methods=['POST'])
def serve_model():
  try:
    print("entrou")
    # request_data = request.get_json(force=True)
    # img = request_data['img']
    # file = request.files['image'].read()
    file = Image.open(BytesIO(request.files['image'].read()))

    print("file")
    # img = np.array(img).reshape(-1, 224, 224, 3)
    # print("reshape")
    # x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    img = tf.keras.preprocessing.image.load_img(file)
    print("passou aki")
    x = tf.keras.preprocessing.image.img_to_array(img)
    print("passou aki2")
    x = np.expand_dims(x, axis=0)
    print("passou aki3")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    print("rodou x")
    preds = model.predict(x)
    print("predicted")
    label_map = {'dangerously deep': 0,
    'feet-dont-touch deep': 1,
    'knee deep': 2,
    'waist deep': 3}
    for pred, value in label_map.items():    
        if value == np.argmax(preds):
            print('Predicted class is:', pred)
            print('With a confidence score of: ', np.max(preds))
            return str(pred) +" "+ str(np.max(preds))

    
  except Exception as e:
    print(e)
    return "An exception occurred"

 
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)