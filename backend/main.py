from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS from flask_cors
from model import Model

try:
    model_instance = Model()
    print("Model has been loaded successfully")
except KeyError:
    print(f'Error:{KeyError}')



app = Flask(__name__)
CORS(app, origins=["*"])



@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/question', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')

        # Use the model instance to generate an answer
        answer = model_instance.answer_question(question)

        return jsonify({'answer': answer})
    except KeyError as e:
        return jsonify({'error': f"KeyError: {str(e)}"}), 404
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)