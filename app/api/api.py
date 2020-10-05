from flask import Blueprint, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from .. import config
from ..service import PhotoService
import json

api = Blueprint('api', __name__, url_prefix='/api/', template_folder="")


@api.route('/', )
def index():
    return render_template('index.html')


@api.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


@api.route('/render-photo', methods=['POST'])
@cross_origin()
def render_photo():
    body = request.json
    # check if fields exist in request
    if 'url' not in body or body['url'] is None:
        return jsonify(error="Missing required parameter 'url'"), 400

    if body['dimensions'] is None:
        return jsonify(error="Missing required object 'dimensions'"), 400

    preview_size = None
    if 'previewSize' in body and body['previewSize'] is not None:
        preview_size = (int(body['previewSize']), int(body['previewSize']))

    scale = 1
    if 'scale' in body and body['scale'] is not None:
        scale = body['scale']

    debug = body['debug']
    image_url = body['url']
    remove_bg_result = {}
    hue = body['hue'] if 'hue' in body and body['hue'] else 'color'
    corner = body['corner'] if 'corner' in body and body['corner'] else 0

    uid = None
    if 'uid' in body and body['uid']:
        uid = body['uid']

    # set client host
    PhotoService.host = request.headers.get('Origin')
    # if no uid means it's new photo
    # let's remove background from image
    if uid is None:
        # remove background from photo
        remove_bg_result = PhotoService.remove_photo_bg(image_url=body['url'])
        print(remove_bg_result)
        image_url = remove_bg_result['url']
    # create instance of service
    # this service responsible for image manipulation
    photo_service = PhotoService(image_url=image_url, dimensions=body['dimensions'], debug=debug or False)
    # auto align face on photo
    photo_service.auto_align_face()
    # find face
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
    # detect face landmark
    photo_service.detect_landmarks(faces[0])

    # adjust original photo according to document standard
    d = body['dimensions']
    photo_service.generate_photo_with_size(int(d['width']), int(d['height']), int(d['crown']), int(d['chin']))

    # if no preview size - save generated photo as final result
    if preview_size is None and uid:
        ext = body['ext'] if 'ext' in body else config.DEFAULT_PHOTO_EXT
        response = photo_service.save_generated_photo(uid=uid, hue=hue, corner=corner, scale=scale, ext=ext)
    else:
        # add preview image as base64 string to response dictionary
        remove_bg_result['base64'] = photo_service.get_result(size=preview_size).decode('ascii')
        response = remove_bg_result

    return jsonify(error="", result=response), 200


@api.route('/save-photo-b64', methods=['POST'])
@cross_origin()
def save_base64_image():
    body = request.json
    # check if fields exist in request
    if 'b64' not in body or body['b64'] is None:
        return jsonify(error="Missing required parameter 'b64'"), 400

    if 'uid' not in body or body['uid'] is None:
        return jsonify(error="Missing required parameter 'uid'"), 400

    hue = body['hue'] if 'hue' in body and body['hue'] else 'color'
    corner = body['corner'] if 'corner' in body and body['corner'] else 0

    host = request.headers.get('Origin')
    uid = body['uid']
    ext = body['ext'] if 'ext' in body else config.DEFAULT_PHOTO_EXT
    size = body['size'] if 'size' in body else None
    # save base64 image to local storage
    result = PhotoService.save_base64_to_image(body['b64'], host, uid, hue, corner, ext=ext, size=size)
    return jsonify(error="", result=result), 200


@api.route('/remove-bg', methods=['POST'])
@cross_origin()
def remove_background():
    body = request.json
    print(body)

    if not body or 'url' not in body or body['url'] is None:
        return jsonify(error="Missing required parameter 'url'"), 400
    PhotoService.remove_photo_bg(body['url'])
    return jsonify(error="", result='success'), 200


@api.route('/watermark', methods=['POST'])
@cross_origin()
def watermark():
    body = request.json
    print(body)
    print(request.headers.get('Origin'))

    watermark_text = 'Demo'
    if request.headers.get('Origin'):
        watermark_text = request.headers.get('Origin').replace('https://', '').replace('http://', '')

    uid = PhotoService.remove_photo_bg(body['url'])
    result = PhotoService.add_watermark(uid, text=watermark_text)
    return jsonify(error="", result=result), 200
