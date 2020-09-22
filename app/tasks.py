import celery as celery
from flask import current_app


@celery.task(bind=True)
def do_some_stuff():
    current_app.logger.info("I have the application context")
