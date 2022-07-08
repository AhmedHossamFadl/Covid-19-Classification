import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import re,base64
from io import BytesIO
from vgg_pytorch import VGG

model = VGG.from_name("vgg16")
model.load_state_dict(torch.load('Trained_Covid-19_Model.pth'))
model.eval()

def base64_to_pil(img_base64):
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image

def Classify_Image(Image):
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image)
    input_batch = input_tensor.unsqueeze(0)

    labels_map = ['Covid19','Normal']

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        logits = model(input_batch)
    preds = torch.topk(logits, k=2).indices.squeeze(0).tolist()

    outputs=[]
    for idx in preds:
        label = labels_map[idx]
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        outputs.append((label,prob))
   
    if outputs[0][1] > outputs[1][1]:
        return outputs[0]
    else:
        return outputs[1]


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classifiy', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        result= Classify_Image(img)
        final_result = 'Class= '+result[0] +', Probability= '+str(round(result[1],2))
        return jsonify(result=final_result)
    return None

@app.route('/classifiy_api', methods=['GET', 'POST'])
def classifiy_api():
    if request.method == 'POST':
        img = base64_to_pil(json.loads(request.json)['image'])
        result= Classify_Image(img)
        return jsonify(class_name=result[0],class_probability=str(round(result[1],2)))
    return None

if __name__ == '__main__':
    app.run()
