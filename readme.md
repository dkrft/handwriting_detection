# Installation

manjaro:

```bash
sudo pacman -S python-tensorflow # works for python 3.7
```

# Heatmap Generation

Slides the CNN over the input image like a convolution and returns a heatmap

```python
cd src
python heatmap.py
```

# TODO

create script, that:
- Fills data/labels with the label masks
- Fills data/raw with the raw images
filenames in such a way, that they are the same in labels and raw for each picture.

so that I can optimize the preprocessor
