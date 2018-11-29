class Postprocessor():
    """postprocessor base class to inherit from"""
    
    def filter(self, img):
        raise NotImplementedError()
        return img