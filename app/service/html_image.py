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


def create_collage(url, uid, host):
    url = 'https://pechat.photo/index.php?route=photoid/photoid/collag123e'
    path = get_tmp_file_path('1234', 'out.jpg')
    print(path)
    imgkit.from_url(url, path, options=options)
    file_name = '{}.{}'.format(config.RESULT_PHOTO_NAME, config.DEFAULT_PHOTO_EXT)
    send_file_over_http(host=host, file_path=path, uid=uid, photo_name=file_name, remove_tmp_path=False)
