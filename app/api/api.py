from flask import Blueprint, render_template

api = Blueprint('api', __name__, url_prefix='/api/', template_folder="")


@api.route('/', )
def index():
    return render_template('index.html')


@api.route('/render-photo', )
def render_photo():
    return "render"
