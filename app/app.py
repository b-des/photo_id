import os
from http import HTTPStatus
from .api import api
from .frontend import frontend
from flask import Flask, render_template, abort, jsonify

BLUEPRINTS = [api, frontend]


def create_app(config=None, app_name=__name__):
    app = Flask(
        app_name,
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
        template_folder="templates",
    )

    app.config.from_object("app.config")

    blueprints = BLUEPRINTS
    blueprints_fabrics(app, blueprints)
    error_pages_fabrics(app)

    return app


def blueprints_fabrics(app, blueprints):
    """Configure blueprints"""

    for blueprint in blueprints:
        app.register_blueprint(blueprint)


def error_pages_fabrics(app):
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
