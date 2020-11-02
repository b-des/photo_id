import logging
import imgkit
import base64
import urllib.parse
from app import config
from ..utils import get_tmp_file_path, send_file_over_http

options = {
    'window-status': 'ready',
    'quiet': '',
    'quality': 100,
    'images': '',
    'zoom': 1,
    'format': 'jpg',
    'height': 760,
    'width': 510
}

logger = logging.getLogger(config.LOGGER_NAME)


def create_collage(uid, host, dimensions):
    url = config.COLLAGE_TEMPLATE_URL.format(host, uid, urllib.parse.urlencode(dimensions))
    logger.info("Generate collage, dimensions: %s, URL: %s", dimensions, url)
    file_name = '{}.{}'.format(config.RESULT_COLLAGE_NAME, config.DEFAULT_PHOTO_EXT)
    path = get_tmp_file_path(uid, file_name)
    imgkit.from_url(url, path, options=options)
    send_file_over_http(host=host, file_path=path, uid=uid, photo_name=file_name)
