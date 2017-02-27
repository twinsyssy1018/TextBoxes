from flask import Flask, request, Response
from flask_cors import CORS
from jsonschema import validate
import json


def serve(func, api_type, host='0.0.0.0', port=3000):
    app = Flask(__name__)
    CORS(app, max_age=3600)
    schema = None
    with open('sdk_python/%s.schema.json' % api_type, 'rw') as f:
        schema = json.loads(f.read())

    @app.route('/predict_multi', methods=['POST'])
    def predict_multi():
        input = request.files['image'].read()
        boxes, image = func(input)
        validate(boxes, schema)
        boxes = json.dumps(boxes)
        boundary = 'AaB0123XX123'
        def gen_response(items, boundary=boundary):
            for i in items:
                res = "--%s\r\nContent-type: %s\r\n\r\n" % (boundary, i[1])
                res += "X"
                res += "\r\n"
                yield res
        return Response(
            gen_response([(boxes, 'application/json'), (image, 'image/jpeg')]),
            mimetype='multipart/mixed; boundary=%s' % boundary)
    @app.route('/predict', methods=['POST'])
    def predict():
        input = request.files['image'].read()
        boxes, image = func(input)
        validate(boxes, schema)
        boxes = json.dumps(boxes)
        return Response(
            image,
            mimetype='image/jpeg')


    app.run(host=host, port=port, threaded=False)

