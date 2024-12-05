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
    
def parse_iphone_data(df):
    data = []

    for _, row in df.iterrows():

        model_info = row['Model']

        series = int(''.join(filter(str.isdigit, model_info.split()[1])))

        price = row['Price']


        storage = int(''.join(filter(str.isdigit, model_info.split('GB')[0].split()[-1])))
        if storage == 1:  # Handle 1TB case
            storage = 1024

        # Determine model type
        is_plus = "Plus" in model_info
        is_pro = "Pro" in model_info and "Max" not in model_info
        is_pro_max = "Pro Max" in model_info
        is_regular = not any([is_plus, is_pro, is_pro_max])

        data.append({
            "series": series,
            "price": price,
            "storage_value": storage,
            "model_Plus": is_plus,
            "model_Pro": is_pro,
            "model_Pro_Max": is_pro_max,
            "model_Regular": is_regular
        })

    return pd.DataFrame(data)


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

            transformed_df = parse_iphone_data(df)
            transformed_df = transformed_df.sort_index()
            df = transformed_df


            df = df[['series', 'storage_value', 'model_Plus', 'model_Pro', 'model_Pro_Max', 'model_Regular']]
            
            # Load model
            model = load_model()
            if model is None:
                return jsonify({'error': 'Không thể load model'}), 500
            
            # Thực hiện dự đoán
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