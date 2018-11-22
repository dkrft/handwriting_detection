class Preprocessor():
    """preprocessor base class to inherit from"""
    
    def preprocess(self, img):
        raise NotImplementedError()
        return img