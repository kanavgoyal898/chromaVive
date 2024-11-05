from flask import Flask, render_template, url_for, redirect, send_file, request, session
from flask_session import Session
import io
import os
import base64

app = Flask(__name__)
app.secret_key = '0123456789'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
app.config['SESSION_PERMANENT'] = False
Session(app)

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
    return render_template('result.html', encoded_image=encoded_image, file_extension=file_extension)

@app.route('/download')
def download():
    encoded_image = session.get('uploaded_image')
    file_extension = session.get('file_extension', 'jpg')
    if encoded_image:
        image_data = base64.b64decode(encoded_image)
        mime_type = 'image/jpeg' if file_extension in ['jpg', 'jpeg'] else 'image/png'
        filename = f"downloaded_image.{file_extension}"
        return send_file(io.BytesIO(image_data), mimetype=mime_type, as_attachment=True, download_name=filename)
    else:
        return redirect(url_for('result'))

if __name__ == '__main__':
    app.run(debug=True)
