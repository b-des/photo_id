import logging
import imgkit

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
    url = config.COLLAGE_TEMPLATE_URL.format(host, uid)
    file_name = '{}.{}'.format(config.RESULT_COLLAGE_NAME, config.DEFAULT_PHOTO_EXT)
    logger.info("Generate collage, URL: %s", dimensions)
    path = get_tmp_file_path(uid, file_name)
    imgkit.from_url(url, path, options=options)
    send_file_over_http(host=host, file_path=path, uid=uid, photo_name=file_name)
