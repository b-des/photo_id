from flask import Blueprint, render_template, redirect

frontend = Blueprint('frontend', __name__, url_prefix='/', template_folder="")


@frontend.route('/', )
def index():
    return render_template('index.html')


@frontend.route('/favicon.ico')
def favicon():
    return redirect('/static/favicon.png')

