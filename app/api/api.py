from flask import Blueprint, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from .. import config
from ..service import PhotoService
import json

api = Blueprint('api', __name__, url_prefix='/api/', template_folder="")


@api.route('/', )
def index():
    return render_template('index.html')


@api.route('/render-photo', methods=['POST'])
@cross_origin()
def render_photo():
    body = request.json
    print(body)
    if 'url' not in body or body['url'] is None:
        return jsonify(error="Missing required parameter 'url'"), 400

    if body['dimensions'] is None:
        return jsonify(error="Missing required object 'dimensions'"), 400

    photo_service = PhotoService(body['url'], dimensions=body['dimensions'])
    photo_service.auto_align_face()
    faces = photo_service.detect_face()
    # if can't find face or found more than one face
    # try again using morphology transformation(e.g. smoothes small objects)
    if len(faces) == 0 or len(faces) > 1:
        print("Can't detect exact face, trying again with morphology transformation")
        faces = photo_service.detect_face(use_morphology=True)

    if len(faces) == 0:
        return jsonify(error="no_face"), 200
    elif len(faces) > 1:
        return jsonify(error="more_one_faces"), 200
    photo_service.detect_landmarks(faces[0])
    d = body['dimensions']
    photo_service.generate_photo_with_size(d['width'], d['height'],
                                           int(d['crown']), int(d['chin']))
    preview_size = None
    if 'previewSize' in body and body['previewSize'] is not None:
        preview_size = (int(body['previewSize']), int(body['previewSize']))

    return jsonify(error="", result=photo_service.get_result(size=preview_size).decode('ascii')), 200


@api.route('/save-photo-b64', methods=['POST'])
@cross_origin()
def save_base64_image():
    body = request.json
    print(body)
    if 'b64' not in body or body['b64'] is None:
        return jsonify(error="Missing required parameter 'b64'"), 400

    PhotoService.save_base64_to_image(body['b64'])
    return jsonify(error="", result='success'), 200
