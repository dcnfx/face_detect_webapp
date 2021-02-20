import base64
import os

# Flask
import cv2
from flask import Flask, request, render_template, jsonify
import requests
from gevent.pywsgi import WSGIServer
import numpy as np
from mtcnn_inference import MtCnn

# Declare a flask app
app = Flask(__name__)

# 初始化模型
mtcnn_c = MtCnn()


def mtcnn_inference(img):
    bounding_boxes, kpoint = mtcnn_c.predict(img)
    confidence = bounding_boxes[:, -1]
    kpoint = kpoint.T
    face_count = len(bounding_boxes)
    face_res = []
    for i in range(face_count):
        res = {"bounding_boxes": {"x1": int(bounding_boxes[i, 0]), "y1": int(bounding_boxes[i, 1]),
                                  "x2": int(bounding_boxes[i, 2]), "y2": int(bounding_boxes[i, 3])},
               "confidence": "%.3f" % confidence[i],
               "kpoint": np.reshape(kpoint[i], (2, 5)).transpose((1, 0)).astype(np.int32).tolist()}
        face_res.append(res)
    result = {"faces": face_res}
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/', methods=['POST'])
def mtcnn():
    if request.form["mode"] == "base64":
        file_binary = base64.b64decode(request.form["base64"].split(",")[1])
    if request.form["mode"] == "image_url":
        image_url = request.form["image_url"]
        file_binary = requests.get(image_url).content
    if request.form["mode"] == "file_path":
        file_path = request.form["file_path"]
        file_path = file_path.strip('/')
        with open(os.path.join(file_path), "rb") as f:
            file_binary = f.read()
    nparr = np.fromstring(file_binary, dtype=np.uint8)
    img = cv2.imdecode(nparr, 1)[:, :, ::-1]
    # inference with one command line
    res = mtcnn_inference(img)

    # Serialize the result, you can add additional fields
    return jsonify(data=res, code=0)


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
