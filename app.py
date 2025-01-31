from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the pre-trained model
def load_model():
    try:
        with open('Model/predictor.pickle', 'rb') as file:
            model = pickle.load(file)
            print(f"Model loaded successfully. Expected number of features: {model.n_features_in_}")
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route("/", methods=['POST', 'GET'])
def index():
    pred_value = None
    if request.method == 'POST':
        try:
            # Fetch inputs
            ram = int(request.form['ram'])
            weight = float(request.form['weight'])
            touchscreen = 1 if request.form.get('touchscreen') == 'yes' else 0
            ips = 1 if request.form.get('ips') == 'yes' else 0

            # Define categorical features
            company_list = ['acer', 'apple', 'asus', 'dell', 'hp', 'lenovo', 'msi', 'other','toshiba']
            type_list = ['2in1Convertible', 'gaming', 'netbook', 'notebook', 'ultrabook', 'workstation']
            os_list = ['linux', 'mac', 'other','windows']
            cpu_list = ['amd', 'intelcorei3', 'intelcorei5', 'intelcorei7', 'other']
            gpu_list = ['amd','intel', 'nvidia']

            # Debugging: Print raw inputs
            print(f"Inputs: ram={ram}, weight={weight}, touchscreen={touchscreen}, ips={ips}")
            print(f"Company: {request.form['company']}, Type: {request.form['type']}, OS: {request.form['os']}")
            print(f"CPU: {request.form['cpu']}, GPU: {request.form['gpu']}")

            # Start feature encoding
            feature_list = [ram, weight, touchscreen, ips]

            # Helper to encode categorical features
            def encode_feature(options, value):
                return [1 if option == value else 0 for option in options]

            feature_list.extend(encode_feature(company_list, request.form['company']))
            feature_list.extend(encode_feature(type_list, request.form['type']))
            feature_list.extend(encode_feature(os_list, request.form['os']))
            feature_list.extend(encode_feature(cpu_list, request.form['cpu']))
            feature_list.extend(encode_feature(gpu_list, request.form['gpu']))

            # Debugging: Print feature list and length
            print(f"Feature List: {feature_list}")
            print(f"Feature List Length: {len(feature_list)}")

            # Prediction
            if model:
                pred_value = model.predict([feature_list])[0] * 300.67  
                pred_value = round(pred_value, 2)
            else:
                pred_value = "Model not loaded"
        except Exception as e:
            print(f"Error: {e}")
            pred_value = "Error processing input. Check input values."

    return render_template('index.html', pred_value=pred_value)

if __name__ == '__main__':
    app.run(debug=True)
