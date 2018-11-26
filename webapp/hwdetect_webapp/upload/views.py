# django stuff
from django.template import Context
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

# image stuff
import numpy as np
import cv2
import base64

# our module
import hwdetect

from hwdetect.utils import show


def create_heat_map(img):

    heatmap = hwdetect.visualization.create_heat_map(img,
            preprocessors=[hwdetect.preprocessor.Bandpass()],
            sampler=hwdetect.visualization.sampler.RandomGrid(),
            predictor=hwdetect.neural_network.Predictor(),
            interpolator=hwdetect.visualization.interpolation.NearestNeighbour())

    # make sure it's 3 channel
    if len(heatmap.shape) == 2:
        heatmap = np.concatenate((heatmap[:,:,None],
                                  heatmap[:,:,None],
                                  heatmap[:,:,None]), axis=2)

    # and between 0 and 255 with dtype utint8
    if heatmap.max() <= 1:
        heatmap *= 255
    heatmap = heatmap.astype(np.uint8)

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

        heatmap = create_heat_map(img)

        print('hmp:', heatmap.shape, heatmap.min(), heatmap.max(), heatmap.dtype)
        print('img:', img.shape, img.min(), img.max(), img.dtype)

        html_base64 = numpy_to_img_base64(heatmap)
 
        context['base64heatmap'] = html_base64

    else:
        # show one white pixel in case no image was uploaded
        html_base64 = numpy_to_img_base64(np.array([[[255,255,255]]]))
        context['base64heatmap'] = html_base64

    if not request.user.is_authenticated:
        print('not authenticated!')
        # https://stackoverflow.com/questions/11241668/what-is-reverse-in-django
        return HttpResponseRedirect(reverse('login'))

    print('authenticated.')

    return render(request, 'interface/interface.html', context)
