import requests
import json
import base64
import sys
from io import BytesIO
from PIL import Image

url = 'http://127.0.0.1:5000/classifiy_api'

image=Image.open(sys.argv[1])
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_64 = base64.b64encode(buffered.getvalue())
result = requests.post(url, json=json.dumps({"image": image_64.decode("utf8")})).json()

print(f'\nPredicted Class: {result["class_name"]}\nProbability= {result["class_probability"]}\n')
