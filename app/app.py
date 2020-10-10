import os
import logging
import time

from .config import ROOT_DIR, LOGGING_FILE, IS_PROD, LOGGER_NAME
from .api import api
from .frontend import frontend
from .static import static
from flask import Flask, jsonify
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

BLUEPRINTS = [api, frontend, static]

logger = logging.getLogger(LOGGER_NAME)


def create_app(config=None, app_name=__name__):
    app = Flask(
        app_name,
        static_folder=os.path.join(ROOT_DIR, "static"),
        template_folder="templates",

    )

    app.config.from_object("app.config")

    blueprints = BLUEPRINTS
    register_blueprints(app, blueprints)
    register_error_pages(app)
    setup_logger()

    return app


def register_blueprints(app, blueprints):
    """Configure blueprints"""

    for blueprint in blueprints:
        app.register_blueprint(blueprint)


def setup_logger():
    #logging.basicConfig(
    #    #filename=LOGGING_FILE if IS_PROD else None,
    #    level=logging.INFO,
    #    format='%(name)s - %(levelname)s:%(asctime)s - %(message)s'
    #)

    #logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)

    # add a rotating handler
    handler = TimedRotatingFileHandler(LOGGING_FILE,
                                       when="midnight",
                                       backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def register_error_pages(app):
    # HTTP error pages definitions

    @app.errorhandler(403)
    def forbidden_page(error):
        return jsonify(error="Forbidden"), 403

    @app.errorhandler(404)
    def page_not_found(error):
        return jsonify(error="Not found"), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify(error="Method not allowed"), 405

    @app.errorhandler(500)
    def server_error_page(error):
        return jsonify(error="Internal server error"), 500
