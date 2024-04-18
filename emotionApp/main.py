from flask import Flask, request, render_template_string
from model import make_prediction
import os

app = Flask(__name__)

HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Emotion Prediction</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            margin: 40px;
            color: #333;
        }
        .container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #5D3FD3;
            text-align: center;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        textarea {
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
            resize: none;
        }
        button {
            cursor: pointer;
            padding: 10px 15px;
            color: white;
            background-color: #5D3FD3;
            border: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #412a9c;
        }
        .results {
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f0f0;
            border-left: 5px solid #5D3FD3;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enter Text for Emotion Prediction</h1>
        <form action="/predict" method="post">
            <textarea name="sample_text" rows="4" cols="50" placeholder="Type your text here..."></textarea>
            <br>
            <button type="submit">Submit</button>
        </form>

        {% if prediction %}
            <div class="results">
                <h2>Prediction Results</h2>
                <p><strong>Logistic Regression:</strong> {{ prediction['Logistic Regression'] }}</p>
                <p><strong>Naive Bayes:</strong> {{ prediction['Naive Bayes'] }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_FORM)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sample_text = request.form['sample_text']
        prediction = make_prediction(sample_text)
        return render_template_string(HTML_FORM, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
