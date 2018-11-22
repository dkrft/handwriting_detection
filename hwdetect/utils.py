from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def get_path(*paths):
    """
    TODO still being used? what about the default trained model,
    I think we need this to properly access the one stored in our repo

    Only used in our project and not part of the api.

    Returns the path to the module on the computer,
    can be changed later if it becomes a 'pip -e .' installable
    module to whatever fits our needs.

    Parameters
    ----------
    *paths : string
        positional parameters for paths to join
        onto the modules path.
        Subdirectories or files relative to the module
    """
    return str(Path(__file__).absolute().parent.parent.joinpath(*paths))
    

def show(img, heat_map=None):
    """
    Opens a window showing any image.

    Parameters
    ----------
    img : array of shape (x, y, 3) or (x, y)
        grayscale or RGB image. width and height
        can be arbitrary
    heat_map : array of shape (x, y, 3) or (x, y)
        same as img. will plot that as well when provided
        and also rescale it to the dimensions of img.

    """

    height, width = img.shape[:2]
    plt.imshow(img)
    if not heat_map is None:
        plt.imshow(cv2.resize(heat_map, (width, height), interpolation=cv2.INTER_NEAREST),
                cmap=plt.cm.viridis,
                alpha=.6,
                interpolation='bilinear')
    plt.xticks([])
    plt.yticks([])
    plt.show()

