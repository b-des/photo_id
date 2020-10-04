import os

from flask import Blueprint

from app.config import ROOT_DIR

static = Blueprint('static', __name__, url_prefix='/static', static_folder=os.path.join(ROOT_DIR, "tmp"))


@static.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
