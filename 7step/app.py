from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
from pytorch_pretrained_vit import ViT
import json

app = Flask(__name__)
app.config['MODEL'] = ViT('B_16_imagenet1k', pretrained=True) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['post'])
def upload():
    img_file = request.files['img_file']
    filename = secure_filename(img_file.filename)
    ul_path = f'./static/{filename}'
    img_file.save(ul_path)
    return render_template('index.html', img_url=ul_path)

@app.route('/recognition', methods=['post'])
def recognition():
    model = app.config['MODEL']
    try:
        img_path = request.form['img_path']
        # img = Image.open(img_path)
        # アルファ値（透過度）を持っている画像に対応
        img = Image.open(img_path).convert('RGB')
        tfms = transforms.Compose([
        transforms.Resize(model.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5],)
])
        img = tfms(img)
    except:
        return render_template('index.html', img_url=img_path, pred_label='err:サポートされている形式の画像ではありません。')
    img = img.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img).squeeze(0)
    pred = torch.argmax(outputs)
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[key] for key in labels_map]
    pred_label = labels_map[pred]
    return render_template('index.html', img_url=img_path, pred_label=pred_label)


if __name__ == "__main__":
    app.run(debug=True)