import cv2
from hwdetect.utils import show

class Scale():
    
    def __init__(self, max_width=2000, max_height=3000):
        """makes sure the image is within the specified bounds
        
        Parameters
        ----------
        max_width : int
            maximum width that the image is allowed to have.
            Will shrink the image if larger
        max_height : int
            maximum height that the image is allowed to have.
            Will shrink the image if larger
            
        """
        self.max_width = max_width
        self.max_height = max_height

    def filter(self, img):
        
        h, w = img.shape[:2]

        max_width = self.max_width
        max_height = self.max_height

        if h > max_height:
            scaling = max_height / h
            w = w * scaling
            h = h * scaling

        if w > max_width:
            scaling = max_width / w
            w = w * scaling
            h = h * scaling

        w = int(w)
        h = int(h)

        img = cv2.resize(img, (w, h))

        return img