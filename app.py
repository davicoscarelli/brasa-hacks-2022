from flask import Flask, request
import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub
import os
from openvino.runtime import Core


model = tf.keras.models.load_model('model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
ie = Core()

model = ie.read_model(model="road-segmentation-adas-0001.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

print("loaded model")

app = Flask(__name__)
@app.route("/model", methods=['POST'])
def serve_model():
  try:
    print("entrou")
    request_data = request.get_json(force=True)
    img = request_data['img']
    
    result = compiled_model([img])[output_layer_ir]
    print( np.argmax(result, axis=1).sum())
    # img = np.array(img).reshape(-1, 224, 224, 3)
    # print("reshape")
    # x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    print("rodou x")
    preds = model.predict(img)
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

    
  except:
    return "An exception occurred"

 
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)