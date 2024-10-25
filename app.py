import os

from flask import Flask, request

from sms_predictor import SMSPredictor
from flask import jsonify

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/sms_spam_detection_tkinter/', methods=['POST']) # path of the endpoint. Except only HTTP POST request
def predict_str():
    # the prediction input data in the message body as a JSON payload
    prediction_inout = request.get_json()
    status = dp.predict_classification(prediction_inout)
    return jsonify({'result': status}), 200


dp = SMSPredictor()
# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)