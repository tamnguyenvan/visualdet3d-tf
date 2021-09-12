from .yolostereo3d_detector import Stereo3D


def get_detector(cfg):
    """
    """
    if cfg.detector.name.lower() == 'stereo3d':
        return Stereo3D(cfg.detector)
    
    raise NotImplementedError(cfg.detector.name)