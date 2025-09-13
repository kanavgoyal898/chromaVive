from flask import Flask
from flask import request, session
from flask import render_template, send_file, url_for, redirect

import io
import os
import uuid
import base64
import tempfile

from colorizers import *

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', '1234567890')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
colorizer_eccv16 = eccv16(pretrained=True).eval().to(device)
colorizer_siggraph17 = siggraph17(pretrained=True).eval().to(device)
print(f'colorizers loaded on {device}')

def tensor_to_encoded_image(tensor, file_extension='png'):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    img_array = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format=file_extension)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_ext = file.filename.split('.')[-1].lower()
            temp_filename = f"{uuid.uuid4().hex}.{file_ext}"
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            file.save(temp_path)

            session['uploaded_file'] = temp_path
            session['file_extension'] = file_ext
            return redirect(url_for('result'))

    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    temp_path = session.get('uploaded_file')
    file_extension = session.get('file_extension', 'png')

    if not temp_path or not os.path.exists(temp_path):
        return redirect(url_for('home'))

    img = load_img(temp_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    tens_l_rs = tens_l_rs.to(device)

    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    def save_temp_image(img_tensor, suffix):
        img_str = tensor_to_encoded_image(img_tensor)
        temp_filename = f"{uuid.uuid4().hex}_{suffix}.{file_extension}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        with open(temp_path, "wb") as f:
            f.write(base64.b64decode(img_str))
        return temp_path, img_str

    eccv16_path, out_img_eccv16 = save_temp_image(out_img_eccv16, "eccv16")
    siggraph17_path, out_img_siggraph17 = save_temp_image(out_img_siggraph17, "siggraph17")

    session['out_img_eccv16'] = eccv16_path
    session['out_img_siggraph17'] = siggraph17_path

    return render_template(
        'result.html',
        out_img_eccv16=out_img_eccv16,
        out_img_siggraph17=out_img_siggraph17
    )

@app.route('/download16')
def download16():
    file_path = session.get('out_img_eccv16')
    extension = session.get('file_extension', 'png')
    if file_path and os.path.exists(file_path):
        return send_file(file_path, mimetype=f"image/{extension}", as_attachment=True, download_name=f"eccv16.{extension}")
    return redirect(url_for('result'))
    
@app.route('/download17')
def download17():
    file_path = session.get('out_img_siggraph17')
    extension = session.get('file_extension', 'png')
    if file_path and os.path.exists(file_path):
        return send_file(file_path, mimetype=f"image/{extension}", as_attachment=True, download_name=f"siggraph17.{extension}")
    return redirect(url_for('result'))

if __name__ == '__main__':
    app.run()