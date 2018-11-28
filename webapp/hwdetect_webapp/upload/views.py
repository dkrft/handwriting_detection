#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


# django stuff
from django.template import Context
from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

# image stuff
import numpy as np
import cv2
import base64

# our module
import hwdetect
from hwdetect.utils import show
from sklearn.neighbors import KNeighborsRegressor
from hwdetect.visualization.interpolation import NearestNeighbour
from hwdetect.preprocessor import Bandpass
from hwdetect.neural_network import Predictor
from hwdetect.visualization.sampler import Random, RandomGrid, Stride

# etc
import time

def create_heat_map(img, bounding_box, use_preproc, use_customint,
                    sampling_method, scale):

    # 1. detection

    preprocessors = []
    if use_preproc:
        # use Scale to make sure ridiculously large scans don't break the system
        preprocessors = [hwdetect.preprocessor.Scale(), hwdetect.preprocessor.Bandpass()]

    interpolator = KNeighborsRegressor()
    if use_customint:
        interpolator = NearestNeighbour()    
    
    # the hyperparamters are optimized such that approx 1000 samples are drawn
    # for each sampler on easy2.jpg. scale is used to scale the params in such
    # a way, that they would approximately sample the number specified in
    # sampling_resolution
    sampler = {
        'RandomGrid': RandomGrid(grid_stepsize=int(30/scale), sample_stepsize=1200),
        'Random': Random(sample_stepsize=int(1000/scale)),
        'Stride': Stride(stride=int(30/np.sqrt(scale)))
    }[sampling_method]

    heat_map = hwdetect.visualization.create_heat_map(img,
                preprocessors=preprocessors,
                sampler=sampler,
                predictor=Predictor(gpu=1),
                interpolator=interpolator,
                postprocessors=[])


    # 2. bounding boxes

    if bounding_box:
        result = hwdetect.visualization.bounded_image(img, heat_map)

    # prepare heat_map for displaying instead,
    # so that it can be interpreted as an image
    else:
        # make sure it's 3 channel
        if len(heat_map.shape) == 2:
            heat_map = np.concatenate((heat_map[:,:,None],
                                      heat_map[:,:,None],
                                      heat_map[:,:,None]), axis=2)
        # and between 0 and 255 with dtype utint8
        if heat_map.max() <= 1:
            heat_map *= 255
            
        result = cv2.resize(heat_map.astype(np.uint8), (img.shape[1], img.shape[0]))

    return result


def numpy_to_img_base64(heat_map, ftype='jpg'):
    # in order to show the image on the frontend, the following things have to be done:
    # 1. encode to jpg or png
    # 2. encode those encoded bytes into base64
    # 3. its a bytes string of base64, so make it utf8 for the frontend
    # 4. add the suffix of "data:image/jpg;base64, " in front of it
    jpg_encoded = cv2.imencode('.'+ftype, heat_map)[1]
    base64_bytes = base64.encodebytes(bytes(jpg_encoded))
    base64_utf8 = base64_bytes.decode("utf-8")
    html_base64 = 'data:image/'+ftype+';base64, ' + base64_utf8
    return html_base64


@csrf_exempt
def process_picture(request):

    # 1. REDIRECT on authentication failure
    if not request.user.is_authenticated:
        print('not authenticated!')
        # https://stackoverflow.com/questions/11241668/what-is-reverse-in-django
        return HttpResponseRedirect(reverse('login'))


    # 2. AJAX for post request
    if request.META['REQUEST_METHOD'] == 'POST':

        # print(request.body)
        # print(request.FILES)
        # print(request.POST)
        # print(request.META['CONTENT_TYPE'])

        if not 'image' in request.FILES:
            print('image missing')
            return JsonResponse({'image':''})

        bounding_box = 'bounding_box' in request.POST and request.POST['bounding_box'] in ['true', 'on']
        use_preproc = 'use_preproc' in request.POST and request.POST['use_preproc'] in ['true', 'on']
        use_customint = 'use_customint' in request.POST and request.POST['use_customint'] in ['true', 'on']
        sampling_method = request.POST['sampling_method']
        sampling_res = float(request.POST['sampling_res'])
        
        # https://stackoverflow.com/questions/46624449/load-bytesio-image-with-opencv
        img_bytesio = request.FILES['image'].file
        file_bytes = np.asarray(bytearray(img_bytesio.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        start = time.time()
        heat_map = create_heat_map(img, bounding_box, use_preproc,
                                   use_customint, sampling_method,
                                   sampling_res)
        benchmark = round(time.time() - start, 1)

        html_base64 = numpy_to_img_base64(heat_map)

        # ajax request
        return JsonResponse({'image':html_base64, 'time':benchmark})

    # 3. HTTP RENDER because was not a post request

    # is ajax now, will hide the img tag until first ajax call
    """# show one white pixel in case no image was uploaded
    html_base64 = numpy_to_img_base64(np.array([[[255,255,255]]]))
    context = {}
    context['base64heat_map'] = html_base64"""

    return render(request, 'interface/interface.html') #, context)
