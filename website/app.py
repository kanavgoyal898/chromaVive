from flask import Flask, render_template, url_for, redirect, request, session

app = Flask(__name__)
app.secret_key = '0123456789'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect('/result')
    else:
        return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        return redirect('/result')
    else:
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)