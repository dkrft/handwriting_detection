# django stuff
from django.template import Context
from django.shortcuts import render

# image stuff
import numpy as np
import cv2
import base64

import time


def create_heatmap_placeholder(img):
    # do some funky stuff to be able to show something
    # on the frontend for now without struggling with the
    # old hwdetect interface
    a = img[:,:,1]
    img[:,:,1] = img[:,:,2]
    img[:,:,2] = a
    heatmap = img

    # simulate delay
    time.sleep(2)

    return heatmap


def numpy_to_img_base64(heatmap, ftype='jpg'):
    # in order to show the image on the frontend, the following things have to be done:
    # 1. encode to jpg or png
    # 2. encode those encoded bytes into base64
    # 3. its a bytes string of base64, so make it utf8 for the frontend
    # 4. add the suffix of "data:image/jpg;base64, " in front of it
    jpg_encoded = cv2.imencode('.'+ftype, heatmap)[1]
    base64_bytes = base64.encodebytes(bytes(jpg_encoded))
    base64_utf8 = base64_bytes.decode("utf-8")
    html_base64 = 'data:image/'+ftype+';base64, ' + base64_utf8
    return html_base64


def process_picture(request):

    context = {}

    # if image was posted in request
    if 'image' in request.FILES:
        
        # https://stackoverflow.com/questions/46624449/load-bytesio-image-with-opencv
        img_bytesio = request.FILES['image'].file
        file_bytes = np.asarray(bytearray(img_bytesio.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        heatmap = create_heatmap_placeholder(img)
        html_base64 = numpy_to_png_base64(heatmap)
 
        context['base64heatmap'] = html_base64

    else:
        # show one white pixel in case no image was uploaded
        html_base64 = numpy_to_png_base64(np.array([[[255,255,255]]]))
        context['base64heatmap'] = html_base64


    return render(request, 'interface/interface.html', context)
