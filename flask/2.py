from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


class FlowerResNet:
    def __init__(self, model_path='CNN.params'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.text_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    def _build_model(self):
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    blk.append(Residual(num_channels, num_channels))
            return blk
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 5)
        )

        return net

    def predict(self, image_file, **kwargs):
        try:

            if isinstance(image_file, str):
                image = Image.open(image_file).convert('RGB')
            else:
                image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
            features = self.transform(image)
            features = features.unsqueeze(0)
            features = features.to(self.device)
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            predicted_class = int(predicted.item())
            confidence_score = float(confidence.item())
            predicted_label = self.text_labels[predicted_class]
            all_probabilities = {
                label: float(prob)
                for label, prob in zip(self.text_labels, probabilities[0].cpu().numpy())
            }

            result = {
                "predicted_class": predicted_label,
                "confidence": confidence_score,
                "probabilities": all_probabilities,
            }

            return result

        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}


app = Flask(__name__)
flower_model = FlowerResNet()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '文件提取失败'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件提取失败'}), 400
    if file and allowed_file(file.filename):
        try:
            form_data = {}
            for key in request.form:
                form_data[key] = request.form[key]
            # （重要********）调用模型进行预测（直接使用文件对象，避免保存到磁盘）
            prediction_result = flower_model.predict(file, **form_data)
            if 'error' in prediction_result:
                return jsonify({
                    'status': 'error',
                    'message': prediction_result['error']
                }), 500

            return jsonify({
                'status': 'success',
                'prediction': prediction_result
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'处理文件时发生错误: {str(e)}'
            }), 500

    else:
        return jsonify({
            'error': '图片解析失败'
        }), 400


@app.route('/predict', methods=['POST'])
def api_predict():
    json_data = {}
    if request.content_type == 'application/json':
        json_data = request.get_json() or {}

    if 'file' not in request.files:
        return jsonify({'error': '文件提取失败'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '文件提取失败'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '文件提取失败'}), 400

    try:
        all_params = {}
        for key in request.form:
            all_params[key] = request.form[key]
        all_params.update(json_data)
        prediction_result = flower_model.predict(file, **all_params)
        if 'error' in prediction_result:
            return jsonify({
                'status': 'error',
                'message': prediction_result['error']
            }), 500
        response = {
            'status': 'success',
            'data': {
                'filename': secure_filename(file.filename),
                'prediction_result': prediction_result,
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'预测过程中发生错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    #上一个程序用的port为5000