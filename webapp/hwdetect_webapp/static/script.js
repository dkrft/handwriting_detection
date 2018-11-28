/**
 * @author Tobias B <github.com/sezanzeb>
 */


'use strict'

window.onload = function () {
    document.getElementById('submit').addEventListener('click', function(e) {

        // check for errors and display them
        if (document.getElementById('image').files.length == 0) {
            document.getElementById('time').innerHTML = 'upload image first'
            e.preventDefault()
            return
        }

        document.getElementById('time').innerHTML = ''

        // show the spinner, hide the old heatmap
        document.getElementById('spinnercontainer').className = 'active'
        document.getElementById('result_wrapper').className = ''

        // construct the request to send pictures and values
        let data = new FormData(document.getElementById('uploadform'))

        // perform the AJAX request
        let req = new XMLHttpRequest()
        req.open('POST', '/', true)
        req.onreadystatechange = function () {
            if (req.readyState != 4 || req.status != 200)
                return;

            // the response contains the base64 image created by the django server
            let response = JSON.parse(req.responseText)
            document.getElementById('result').src = response['image']
            document.getElementById('time').innerHTML = response['time'] + ' Seconds'

            // show
            document.getElementById('spinnercontainer').className = ''
            document.getElementById('result_wrapper').className = 'visible'
        }
        req.send(data)

        // don't redirect
        e.preventDefault()
    })
}