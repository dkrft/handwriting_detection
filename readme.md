# Handwriting Detection WebApp

```bash
cd webapp/hwdetect_webapp
python manage.py runserver
```

**superuser:**
- proxima
- admin

then open the browser according to what the console output says.


# Handwriting Detection Offline Use

Draws a heatmap onto an image and shows it. The heatmap shows
where our model suspects handwriting.

It consists of a CNN that classifies chunks from the input image,
which corresponds to checking if in the area of the chunk handwriting
is present. Doing this multiple times on various places on the image
forms the heatmap.

Afterwards, noise is removed. Noise appears very grainy and rather sparse
on the heatmap.

see the jupyter-notebook examples/hwdetect_example.ipynb
on how to use it in your coding project.


## Installation

```bash
pip3 install -e .
```

manjaro:

```bash
sudo pacman -S python-tensorflow # works for python 3.7
```


## Pretrained Models

you can download pretrained models here:
https://www.dropbox.com/sh/jzn0kzsw5a9o4rm/AABovJ6qr5zvLSBxVbGBvVdKa?dl=0

download those into any path, or:

download those into the place where hwdetect is installed into
hwdetect/neural_network/trained_models/, for example

`/usr/lib/python3.7/site-packages/hwdetect/neural_network/trained_models/`

Create the folder if it doesn't exist. You can figure the path to the
module out using the following command in a terminal:

`python -c "import hwdetect; print(hwdetect.__file__)`



## Training

On how to train, see the jupyter-notebook *examples/hwdetect_example.ipynb*

**Data**

Training data can be obtained here:

https://www.dropbox.com/sh/rs7cg4x3q6gx1kx/AAAvdm097o6C3GPitm5SeHYPa?dl=0

You can put those two folders anywhere you like, or put it into the modules
path into hwdetect/data/data_sets/, for example:

`/usr/lib/python3.7/site-packages/hwdetect/data/data_sets/`

Create the subfolder *data_sets/* if it doesn't exist. You can figure the path to the
module out using the following command in a terminal:

`python -c "import hwdetect; print(hwdetect.__file__)`



# Future

- train the model on preprocessed data instead of raw data, if it is going to
predict on the in the same way preprocessed data later anyways
- or: add preprocessed pictures (for example a bandpassed)
into something like another channel of the image which might make
the NN prediction results better because it has more information available
- add parameters to the webapp to play around with the settings
- split input image into line chunks, scale them to equal line height,
then use bandpass preprocessing since after scaling the bandpass
frequencies should match no matter the used font size. After chunking
into lines, deskewing of individual lines might be relatively trivial,
just figure out the 4 edges of the lines as it was a rectangle and from
there calculate the skew angle. So this would make the bandpass preprocessor
perform better.
