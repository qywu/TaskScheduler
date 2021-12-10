from flask import Flask, render_template
from flask_socketio import SocketIO
import pyinotify
import csv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
thread = None

def get_csv():
    return

@app.route("/")
def index():
    template = 'index.html'
    csv_data = get_csv()
    return render_template(template, data = csv_data)

if __name__ == '__main__':
    app.run(debug = True, use_reloader = True)