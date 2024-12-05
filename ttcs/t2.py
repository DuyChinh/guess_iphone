from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục uploads nếu chưa tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model đã train (bạn cần thay thế đường dẫn tới model của bạn)
def load_model():
    try:
        # Tải mô hình bằng joblib
        model = joblib.load('model/random_forest_model.pkl')
                # model = joblib.load('model/random_forest_model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Đọc file Excel
            df = pd.read_excel(filepath)
            
            # Load model
            model = load_model()
            if model is None:
                return jsonify({'error': 'Không thể load model'}), 500
            
            # Thực hiện dự đoán
            features = ['series', 'storage_value', 'model_Plus', 'model_Pro' ,'model_Pro_Max' , 'model_Regular' ]
            input_data = df[features]
            input_data = input_data.fillna(0)  # Thay giá trị NaN bằng 0
            input_data = input_data.values

            predictions = model.predict(df)

            
            # Thêm kết quả dự đoán vào DataFrame
            df['Predicted_Price'] = predictions
            
            # Lưu kết quả vào file Excel mới
            result_filename = 'prediction_result.xlsx'





            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            df.to_excel(result_filepath, index=False)
            
            return jsonify({
                'success': True,
                'message': 'Dự đoán thành công',
                'predictions': predictions.tolist(),
                'result_file': result_filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File không hợp lệ'}), 400

if __name__ == '__main__':
    app.run(debug=True)