'use strict'

/**
 * @author Tobias B <github.com/sezanzeb>
 */

/**
 * hides loading spinner and writes 'Internal Server Error' or the msg parameter into the form
 */
function error(msg) {
    // show that an error occured, reset webpage
    if(msg == undefined)
        msg = 'Internal Server Error'
    document.getElementById('time').innerHTML = msg
    document.getElementById('spinnercontainer').className = ''
}

/**
 * shows the image on the frontend specified by id
 * @param id {string} one of 'result', 'preproc' and 'heat_map', or -1 to hide them
 */
function show(id) {

    let classNames = {
        'result':'',
        'preproc':'',
        'heat_map':'',
    }
    classNames[id] = 'visible'

    document.getElementById('result').className = classNames['result']
    document.getElementById('preproc').className = classNames['preproc']
    document.getElementById('heat_map').className = classNames['heat_map']

    // transition in when the images receive the src base64 for the first time
    // using a wrapper, otherwise it won't transition
    if(id === -1) {
        document.getElementById('img_wrapper').className = ''
        console.log('hide img_wrapper')
    }
    else {
        document.getElementById('img_wrapper').className = 'visible'
        console.log('show img_wrapper')
    }
}

// buttons that show either bounding boxes, heatmap or preprocessing
document.getElementById('result_button').addEventListener('click', function() { show('result') })
document.getElementById('preproc_button').addEventListener('click', function() { show('preproc') })
document.getElementById('heat_map_button').addEventListener('click', function() { show('heat_map') })

window.onload = function () {
    // after the form has been filled in, starts a request on the server
    // to detect handwriting on the provided image
    document.getElementById('submit').addEventListener('click', function(e) {

        // check for errors and display them
        if (document.getElementById('image').files.length == 0) {
            document.getElementById('time').innerHTML = 'upload image first'
            e.preventDefault()
            return
        }

        // document.getElementById('time').innerHTML = ''

        // show the spinner, hide the old images
        document.getElementById('spinnercontainer').className = 'active'
        show(-1)

        // construct the request to send pictures and values
        let data = new FormData(document.getElementById('uploadform'))

        // perform the AJAX request for the images
        let req = new XMLHttpRequest()
        req.open('POST', '/', true)
        req.onreadystatechange = function () {
            if (req.status == 500)
                return error()

            if (req.readyState != 4 || req.status != 200)
                return

            // the response contains the base64 image created by the django server
            let response = JSON.parse(req.responseText)
            document.getElementById('result').src = response['result']
            document.getElementById('preproc').src = response['preproc']
            document.getElementById('heat_map').src = response['heat_map']
            document.getElementById('time').innerHTML = response['time'] + ' Seconds'

            // show
            document.getElementById('spinnercontainer').className = ''
            show('result')
        }
        req.send(data)


        // now start showing the progress
        // using a stream
        // https://www.w3.org/TR/eventsource/
        let stream = new EventSource('/')
        stream.onmessage = function(e) {
            let log = e.data
            if(log != "") {
                document.getElementById('console').innerHTML = log
                let console_wrapper = document.getElementById('console_wrapper')
                console_wrapper.scrollTop = console_wrapper.scrollHeight
            }
        }

        // so basically this is just for the console
        // output, nothing dramatic, don't create panic
        // because of something minor. commented out
        /*stream.onerror = function(e) {
            console.log(e)
            error('error, see log')
        }*/

        // and don't redirect to a new page because the browser
        // will do that by default to show the response in plain
        // text or rendered if html
        e.preventDefault()
    })
}



// YYYYEEEEESSSSS THE EVENT STREAM WORKS WITH DJANGO FINALLY
// no long polling and cluttering logs and debug tools with requests anymore
/*// I have no idea how create an EventSource object
// in combination with django, so I do it with long
// polling
function requestProgress(heatmap_request_object) {
    // while working on the request
    if(heatmap_request_object.readyState != 4) {
        
        let req2 = new XMLHttpRequest()
        let data2 = new FormData()
        req2.open('POST', '/', true)
        req2.onreadystatechange = function () {
            if (req2.status == 500)
                return error()

            if (req2.readyState != 4 || req2.status != 200)
                return;

            let response = JSON.parse(req2.responseText)['console']
            // scroll to bottom (super annoying when
            // you want to read old output on top)
            if(response != "") {
                document.getElementById('console').innerHTML = response
                console_wrapper = document.getElementById('console_wrapper')
                console_wrapper.scrollTop = console_wrapper.scrollHeight

                // long poll
                requestProgress(heatmap_request_object)
            }
        }
        // append progress, so that django knows that the log is being requested
        // add the csrf token, so that the request is accepted
        data2.append('progress', 1)
        data2.append('csrfmiddlewaretoken', document.getElementsByName('csrfmiddlewaretoken')[0].value)
        req2.send(data2)
    }
}*/