import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

# rectangular density based clustering
# uses cityblock distance and two epsilons
# one for each dimension
def rdbscan(points, eps_x, eps_y, *args, **kwargs):
    """points is an array of [x1, y1], [x2, y2], ... pairs.
    Returns label id for each point."""

    # do the following thing on a copy and use np arrays:
    points = np.array(points, dtype=float)

    # Distance matrix contains only one
    # value per datapoint pair, so to
    # get a rectangular epsilon, the
    # ratio between eps_x and eps_y needs
    # to be encoded in the distance values.
    # Divide x or y value such that the
    # rectangular cityblock epsilon can
    # be calculated by a squared epsilon.
    eps = 0
    if eps_x > eps_y:
        points[:,0] /= (eps_x / eps_y)
        eps = eps_y
    elif eps_x < eps_y:
        points[:,1] /= (eps_y / eps_x)
        eps = eps_x
    else:
        eps = eps_x

    db = DBSCAN(eps=eps, *args, metric='cityblock', **kwargs).fit(points)
    labels = db.labels_

    return labels


def rotate(img, angle=None):
    """if angle is None, will automatically detect angle"""

    if angle is None: 
        edges = cv2.Canny(img, 50, 150, apertureSize = 3)
        bunch_of_lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

        # ...,1] for theta, ...,0] for rho
        angle = - ((np.pi/2) - (np.median(bunch_of_lines[:,0,1]) % (np.pi/2)))

    #rotation angle in degree
    rotated = ndimage.rotate(img, angle * 360 / np.pi / 2, cval=img.max())

    # crop
    a = int((rotated.shape[0] - img.shape[0])/2)
    b = int((rotated.shape[1] - img.shape[1])/2)
    rotated = rotated[a:-a-1, b:-b-1]

    return rotated, angle


def show(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
