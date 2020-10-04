import os

from flask import Blueprint, render_template, redirect

frontend = Blueprint('frontend', __name__, url_prefix='/', template_folder="")


@frontend.route('/', )
def index():
    return os.environ['REMOVE_BG_API_KEY']#render_template('index.html')


@frontend.route('/favicon.ico')
def favicon():
    return redirect('/static/favicon.png')

