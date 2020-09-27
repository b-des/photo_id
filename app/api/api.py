from flask import Blueprint, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from .. import config
from ..service import PhotoService
import json

api = Blueprint('api', __name__, url_prefix='/api/', template_folder="")


@api.route('/', )
def index():
    return render_template('index.html')


def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj

@api.route('/render-photo', methods=['POST'])
@cross_origin()
def render_photo():
    body = request.json
    print(body)
    if 'url' not in body or body['url'] is None:
        return jsonify(error="Missing required parameter 'url'"), 400
    photo_service = PhotoService(body['url'])
    photo_service.auto_align_face()
    faces = photo_service.detect_face()
    if len(faces) == 0:
        return jsonify(error="no_face"), 200
    elif len(faces) > 1:
        return jsonify(error="more_one_faces"), 200
    photo_service.detect_landmarks(faces[0])
    photo_service.generate_photo_by_params(config.FINAL_PHOTO_WIDTH, config.FINAL_PHOTO_HEIGHT,
                                           config.TOP_HEAD_LINE, config.BOTTOM_HEAD_LINE)
    size = None
    if 'size' in body and body['size'] is not None:
        size = (int(body['size']), int(body['size']))
    print(photo_service.get_result(size=size).decode('ascii'))
    return jsonify(error="", result=photo_service.get_result(size=size).decode('ascii')), 200
