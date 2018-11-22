class Postprocessor():
    """postprocessor base class to inherit from"""
    
    def postprocess(self, img):
        raise NotImplementedError()
        return img