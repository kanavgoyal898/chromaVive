from flask import Flask, render_template, url_for, redirect, send_file, request, session
from flask_session import Session
import io
import base64

from colorizers import *

app = Flask(__name__)
app.secret_key = '0123456789'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
Session(app)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
colorizer_eccv16 = eccv16(pretrained=True).eval().to(device)
colorizer_siggraph17 = siggraph17(pretrained=True).eval().to(device)
print(f'colorizers loaded on {device}')

def tensor_to_encoded_image(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    img_array = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
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
            file_content = file.read()
            encoded_image = base64.b64encode(file_content).decode('utf-8')
            session['uploaded_image'] = encoded_image
            session['file_extension'] = file.filename.split('.')[-1].lower()
            return redirect(url_for('result'))
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    encoded_image = session.get('uploaded_image')
    file_extension = session.get('file_extension', 'jpg')
    image = io.BytesIO(base64.b64decode(encoded_image))
    
    img = load_img(image)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256,256))
    tens_l_rs = tens_l_rs.to(device)
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    img_bw = tensor_to_encoded_image(img_bw)
    out_img_eccv16 = tensor_to_encoded_image(out_img_eccv16)
    out_img_siggraph17 = tensor_to_encoded_image(out_img_siggraph17)
    session['img_bw'] = img_bw
    session['out_img_eccv16'] = out_img_eccv16
    session['out_img_siggraph17'] = out_img_siggraph17
    return render_template('result.html', encoded_image=encoded_image, file_extension=file_extension, img_bw=img_bw, out_img_eccv16=out_img_eccv16, out_img_siggraph17=out_img_siggraph17)

@app.route('/download16')
def download16():
    encoded_image = session.get('out_img_eccv16')
    file_extension = session.get('file_extension', 'jpg')
    if encoded_image:
        image_data = base64.b64decode(encoded_image)
        mime_type = 'image/jpeg' if file_extension in ['jpg', 'jpeg'] else 'image/png'
        filename = f"downloaded_image_16.{file_extension}"
        return send_file(io.BytesIO(image_data), mimetype=mime_type, as_attachment=True, download_name=filename)
    else:
        return redirect(url_for('result'))
    
@app.route('/download17')
def download17():
    encoded_image = session.get('out_img_siggraph17')
    file_extension = session.get('file_extension', 'jpg')
    if encoded_image:
        image_data = base64.b64decode(encoded_image)
        mime_type = 'image/jpeg' if file_extension in ['jpg', 'jpeg'] else 'image/png'
        filename = f"downloaded_image_17.{file_extension}"
        return send_file(io.BytesIO(image_data), mimetype=mime_type, as_attachment=True, download_name=filename)
    else:
        return redirect(url_for('result'))

if __name__ == '__main__':
    app.run()