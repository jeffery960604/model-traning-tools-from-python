from flask import Flask, render_template, request, jsonify
import os
import threading
from finetune.finetune_gguf import start_finetune

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    model_file = request.files['model']
    data_file = request.files['data']
    
    model_path = os.path.join(UPLOAD_FOLDER, model_file.filename)
    data_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    
    model_file.save(model_path)
    data_file.save(data_path)
    
    # Start finetuning in background
    params = {
        'model_path': model_path,
        'data_path': data_path,
        'epochs': int(request.form['epochs']),
        'lr': float(request.form['lr'])
    }
    
    thread = threading.Thread(target=start_finetune, kwargs=params)
    thread.start()
    
    return jsonify({'status': 'training started'})

@app.route('/progress')
def get_progress():
    # Implement progress tracking logic
    return jsonify({'progress': 50, 'status': 'training'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    