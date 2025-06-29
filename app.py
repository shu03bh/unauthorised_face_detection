from flask import Flask, render_template, request, redirect, url_for, flash
from model_utils import register_user, authenticate_user

app = Flask(__name__)
app.secret_key = 'your-secret-key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        success, msg = register_user(name)
        flash(msg, 'success' if success else 'danger')
        return redirect(url_for('index' if success else 'register'))
    return render_template('registration.html')


@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        success, msg = authenticate_user()
        flash(msg, 'success' if success else 'warning')
        return redirect(url_for('index' if success else 'authenticate'))
    return render_template('authentication.html')



if __name__ == '__main__':
    app.run(debug=True)
