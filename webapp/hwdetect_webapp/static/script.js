"use strict"

document.getElementById('form').addEventListener('submit', function() {
    document.getElementById('spinner').className = 'active'
    document.getElementById('heatmap').style.opacity = 0
})

// trigger transition
setTimeout(function() {
    document.getElementById('heatmap').style.opacity = 1
}, 1);