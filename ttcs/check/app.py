from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import glob

app = Flask(__name__)

# Cấu hình thư mục
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'  # Thêm đường dẫn thư mục model
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['CURRENT_MODEL'] = os.path.join(MODEL_FOLDER, 'stacking_model.pkl')  # Model mặc định


for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# # Tạo thư mục uploads nếu chưa tồn tại
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model đã train (bạn cần thay thế đường dẫn tới model của bạn)
def load_model():
    try:
        model_path = app.config.get('CURRENT_MODEL')
        if not os.path.exists(model_path):
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    






@app.route('/get_models')
def get_models():
    """Lấy danh sách các model có sẵn"""
    try:
        model_files = glob.glob(os.path.join(app.config['MODEL_FOLDER'], '*.pkl'))
        models = [os.path.basename(f) for f in model_files]
        current_model = os.path.basename(app.config['CURRENT_MODEL'])
        return jsonify({
            'models': models,
            'current_model': current_model
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select_model', methods=['POST'])
def select_model():
    """Chọn model để sử dụng"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Tên model không hợp lệ'}), 400
            
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model không tồn tại'}), 404
            
        # Kiểm tra xem có load được model không
        try:
            test_model = joblib.load(model_path)
        except Exception as e:
            return jsonify({'error': f'Không thể load model: {str(e)}'}), 500
            
        # Nếu load thành công, cập nhật model hiện tại
        app.config['CURRENT_MODEL'] = model_path
        
        return jsonify({
            'success': True, 
            'message': 'Đã chọn model thành công',
            'model_name': model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Lấy thông tin về model hiện tại"""
    try:
        current_model = os.path.basename(app.config['CURRENT_MODEL'])
        model_path = app.config['CURRENT_MODEL']
        model_size = os.path.getsize(model_path) // 1024  # Kích thước theo KB
        model_modified = os.path.getmtime(model_path)  # Thời gian sửa đổi
        
        return jsonify({
            'name': current_model,
            'size': f'{model_size}KB',
            'last_modified': model_modified,
            'path': model_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500








    
def parse_iphone_data(df):
    data = []

    for _, row in df.iterrows():

        model_info = row['Model']

        series = int(''.join(filter(str.isdigit, model_info.split()[1])))



        storage = int(''.join(filter(str.isdigit, model_info.split('GB')[0].split()[-1])))
        if storage == 1:  # Handle 1TB case
            storage = 1024

        # Determine model type
        is_plus = "Plus" in model_info
        is_pro = "Pro" in model_info and "Max" not in model_info
        is_pro_max = "Pro Max" in model_info
        is_regular = not any([is_plus, is_pro, is_pro_max])
        year = row['Year']
        year_difference = year - df['Year'].min()

        data.append({
            "series": series,
            "storage_value": storage,
            "model_Plus": is_plus,
            "model_Pro": is_pro,
            "model_Pro_Max": is_pro_max,
            "model_Regular": is_regular,
            "year": year

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
            # Đọc và xử lý file Excel
            df = pd.read_excel(filepath)
            transformed_df = parse_iphone_data(df)
            df = transformed_df[['series', 'storage_value', 'model_Plus', 'model_Pro', 
                               'model_Pro_Max', 'model_Regular', 'year']]
            
            # Load model
            model = load_model()
            if model is None:
                return jsonify({'error': 'Không thể load model. Vui lòng kiểm tra lại model đã chọn'}), 500
            
            # Thực hiện dự đoán
            predictions = model.predict(df)
            predictions = predictions.round().astype(int)
            df['Predicted_Price'] = predictions
            
            # Lưu kết quả
            result_filename = f'prediction_result_{os.path.splitext(filename)[0]}.xlsx'
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            df.to_excel(result_filepath, index=False)
            
            return jsonify({
                'success': True,
                'message': 'Dự đoán thành công',
                'predictions': df.to_dict(orient='records'),
                'result_file': result_filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Lỗi khi xử lý: {str(e)}'}), 500
        finally:
            # Xóa file tạm nếu cần
            if os.path.exists(filepath):
                os.remove(filepath)
            
    return jsonify({'error': 'File không hợp lệ'}), 400

if __name__ == '__main__':
    app.run(debug=True)