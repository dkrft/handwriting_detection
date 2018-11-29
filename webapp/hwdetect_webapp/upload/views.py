#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

__author__ = "Tobias B <github.com/sezanzeb>"
__version__ = "1.0"


# django stuff
from django.template import Context
from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
# image stuff
import numpy as np
import cv2
import base64
# our module
import hwdetect
from hwdetect.utils import show, get_path
from sklearn.neighbors import KNeighborsRegressor
from hwdetect.visualization.interpolation import NearestNeighbour
from hwdetect.preprocessor import Bandpass, Threshold, Scale
from hwdetect.neural_network import Predictor
from hwdetect.visualization.sampler import Random, RandomGrid, Stride
# etc
import time
from threading import Thread
import logging
import io
import os


# enable INFO on all loggers in the hwdetect packages
logger = logging.getLogger('hwdetect')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
# and write into StringIO in order to read and display the log
# for the frontend
log_stream = io.StringIO()
logger.addHandler(logging.StreamHandler(log_stream))


# show me info logs of this single view
# (I disabled all other django info logs in
# settings.py)
view_logger = logging.getLogger(__name__)
view_logger.setLevel(logging.INFO)
view_logger_handler = logging.StreamHandler()
view_logger_handler.setFormatter(logging.Formatter('[%(message)s]'))
view_logger.addHandler(view_logger_handler)



def create_heat_map(img, bounding_box, use_preproc, use_customint,
                    sampling_method, scale, path):

    # normalize and threshold
    img = Threshold().filter(img)

    # prepare the preprocessing pipeline if wanted
    preprocessors = []
    if use_preproc:
        # use Scale to make sure ridiculously large scans don't break the system
        preprocessors = [Scale(), Bandpass()]

    # and prepare the interpolator according to the frontend form settings
    interpolator = KNeighborsRegressor()
    if use_customint:
        interpolator = NearestNeighbour()    
    
    # the hyperparamters are optimized such that approx 1000 samples are drawn
    # for each sampler on easy2.jpg. scale is used to scale the params in such
    # a way, that they would approximately sample the number specified in
    # sampling_resolution
    sampler = {
        'RandomGrid': RandomGrid(grid_stepsize=int(np.ceil(30/scale)), sample_stepsize=1200),
        'Random': Random(sample_stepsize=int(np.ceil(1000/scale))),
        'Stride': Stride(stride=int(np.ceil(30/np.sqrt(scale))))
    }[sampling_method]

    # smaller heatmap when scale decreases
    # for faster interpolation
    heat_map_scale = int(max(10, min(20, 10/scale)))

    heat_map, preproc = hwdetect.visualization.create_heat_map(img,
                            preprocessors=preprocessors,
                            sampler=sampler,
                            predictor=Predictor(gpu=1),
                            interpolator=interpolator,
                            postprocessors=[],
                            heat_map_scale=heat_map_scale,
                            return_preprocessed=True)
    hm = heat_map

    # if preprocessing was disabled, then do that now in order
    # to visualize it on the frontend
    if not use_preproc:
        preproc = hwdetect.preprocessor.Bandpass().filter(img)


    # bounding boxes
    result = hwdetect.visualization.bounded_image(img, heat_map)


    # prepare heat_map for displaying
    # so that it can be interpreted as an image
    # make sure it's 3 channel
    heat_map = np.concatenate((heat_map[:,:,None],
                               heat_map[:,:,None],
                               heat_map[:,:,None]),
                               axis=2)
    # 2 is Red, 1 if Green, 0 is Blue
    #                                     dark                    light
    heat_map[:,:,2] = ((1-heat_map[:,:,2])*130 + (heat_map[:,:,2])*125) # /2 + 128
    heat_map[:,:,1] = ((1-heat_map[:,:,1])*35  + (heat_map[:,:,1])*208) # /2 + 128
    heat_map[:,:,0] = ((1-heat_map[:,:,0])*92  + (heat_map[:,:,0])*182) # /2 + 128
    # and between 0 and 255 with dtype utint8
    # (but don't normalize, if all are low probabilities, keep them low)
    # heat_map *= 255
    # and of the same size as the other images
    heat_map = cv2.resize(heat_map.astype(np.uint8), (img.shape[1], img.shape[0]))


    # store in filesystem
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path + 'result.jpg', result)
    cv2.imwrite(path + 'preproc.jpg', preproc)

    hwdetect.visualization.plot_heat_map(img, hm, save_as=path + 'heat_map.jpg')

    # cv2.imwrite(path + 'heat_map.jpg', heat_map)

    view_logger.info('results stored to:"' + path + '"')


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


# @csrf_exempt
def process_picture(request):

    # 1. REDIRECT on authentication failure
    if not request.user.is_authenticated:
        view_logger.info('not authenticated!')
        # https://stackoverflow.com/questions/11241668/what-is-reverse-in-django
        return HttpResponseRedirect(reverse('login'))


    if request.META['HTTP_ACCEPT'] == 'text/event-stream':
        # 3. Progress Report
        view_logger.info('stream')
        
        # wait for changed log_stream
        def answerGenerator():
            previous = ''
            while True:
                answer = log_stream.getvalue()
                # only if log changed
                if answer == previous:
                    time.sleep(0.1)
                else:
                    previous = answer
                    # https://www.w3.org/TR/eventsource/
                    message = 'data: ' + ('\ndata: '.join(answer.split('\n')[:-1])) + '\n\n'
                    yield message

        return StreamingHttpResponse(answerGenerator(), content_type='text/event-stream')
        # return JsonResponse({'console':log_stream.getvalue()})


    if request.META['REQUEST_METHOD'] == 'POST':
        
        view_logger.info('request for heat_map')
        # 2. AJAX for post request

        # view_logger.info(request.body)
        # view_logger.info(request.FILES)
        # view_logger.info(request.POST)
        # view_logger.info(request.META['CONTENT_TYPE'])

        if not 'image' in request.FILES:
            view_logger.info('image missing')
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
        # do it in a separate process so that status updates
        # can be sent to the frontend in the meantime
        path = time.strftime("%y.%m.%d_%H.%M.%S.") + str(int(time.time()*100%100)) + '/'
        path = get_path('webapp/results/', path) + '/'
        t = Thread(target=create_heat_map,
                   args=(img, bounding_box, use_preproc,
                         use_customint, sampling_method,
                         sampling_res, path))
        t.start()
        t.join()
        benchmark = round(time.time() - start, 1)

        # clear log
        log_stream.truncate(0)
        log_stream.seek(0)

        # queue is too small for the image
        # store in filesystem and read from that
        result = cv2.imread(path + 'result.jpg')
        preproc = cv2.imread(path + 'preproc.jpg')
        heat_map = cv2.imread(path + 'heat_map.jpg')

        # happens when image failed to write in create_heat_map
        assert not result is None
        assert not preproc is None
        assert not heat_map is None

        # serve result properly in ajax request as image
        return JsonResponse({
            'result':numpy_to_img_base64(result),
            'preproc':numpy_to_img_base64(preproc),
            'heat_map':numpy_to_img_base64(heat_map),
            'time':benchmark
        })


    # 4. HTTP RENDER because was not a post request and no stream request

    # is ajax now, will hide the img tag until first ajax call
    """# show one white pixel in case no image was uploaded
    html_base64 = numpy_to_img_base64(np.array([[[255,255,255]]]))
    context = {}
    context['base64heat_map'] = html_base64"""

    view_logger.info('request for interface.html')

    return render(request, 'interface/interface.html') #, context)
