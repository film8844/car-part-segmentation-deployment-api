import os

SECRET_KEY = '7e4d6d646e788917047ce3298ab34dcec724772faaa9c7dd'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
UPLOAD_FOLDER_DAMAGE  = os.path.join('static', 'uploads_damage')
if not os.path.exists(UPLOAD_FOLDER_DAMAGE):
    os.mkdir(UPLOAD_FOLDER_DAMAGE)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

PORT = 3000
