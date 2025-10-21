from flask import Flask, request, jsonify
app = Flask(__name__)
def get_user_id():
    info ={}
    with open('id.txt', 'r', encoding='utf-8') as f:
       for line in f:
            line = line.strip()
            token,name = line.split(',')
            info[token] = name
    return info

@app.route('/lr', methods=['POST'])
def lr():
    token = request.args.get('token')
    if not token:
        return("认证失败")
    user_id=get_user_id()
    if token not in user_id:
        return ("认证失败")
    print("user_id",user_id[token])
    data = request.get_json()
    if not data:
        return jsonify({"error": "解析失败"}), 400
    Operation = data.get('Operation')
    Data_A = data.get('Data_A')
    Data_B = data.get('Data_B')
    if Operation is None or Data_A is None or Data_B is None:
        return jsonify({"error": "解析失败"}), 400

    try:
        Data_A = float(Data_A)
        Data_B = float(Data_B)
    except (ValueError, TypeError):
        return jsonify({"error": "解析失败"}), 400
    if Operation == 'sum':
        result = Data_A + Data_B
        return jsonify({'Result': result})
    elif Operation == 'subtract':
        result = Data_A - Data_B
        return jsonify({'Result': result})
    elif Operation == 'multiply':
        result = Data_A * Data_B
        return jsonify({'Result': result})
    elif Operation == 'divide':
        if Data_B == 0:
            return jsonify({"error": "除数不能为零"}), 400
        result = Data_A / Data_B
        return jsonify({'Result': result})
    else:
        return jsonify({"error": "模型预测失败"}), 400
if __name__ == '__main__':
    app.run()
