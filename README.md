# Handwriting Detection

Draws a heatmap onto an image and shows it. The heatmap shows
where our model suspects handwriting.

It consists of a CNN that classifies chunks from the input image,
which corresponds to checking if in the area of the chunk handwriting
is present. Doing this multiple times on various places on the image
forms the heatmap.

Afterwards, noise is removed. Noise appears very grainy and rather sparse
on the heatmap.

### Installation

pip dependencies:

```
scikit-learn>=0.20.0
opencv-python>=3.4.3.18
tensorflow>=1.11.0
```

manjaro:

```bash
sudo pacman -S python-tensorflow # works for python 3.7
```

### Usage

Right now the code points to an image 'training_data/test_2.jpg'
for testing purposes. Make sure this image exists, which can be
any scanned document with handwriting on it.

```python
python3 src/HeatMap.py
```

### Files

**HWClassifier.py**

File containing functionality to construct and train a convolution neural network for handwriting recognition.

**HWInterface.py**

Interface to easily query a trained cnn.

**LogAnalyzer.py**

File containing functionality to analyze and visualize the training process of the cnn.
