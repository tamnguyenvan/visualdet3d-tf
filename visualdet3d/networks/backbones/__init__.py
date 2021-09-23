from visualdet3d.networks.utils.registry import BACKBONE_DICT


def build_backbone(cfg):
    temp_cfg = cfg.copy()
    name = ""
    if 'name' in temp_cfg:
        name = temp_cfg.pop('name')
    else:
        name = 'resnet'

    return BACKBONE_DICT[name](**temp_cfg)