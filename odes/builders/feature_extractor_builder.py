from odes.core.feature_extractors.bev_vgg import BevVgg
from odes.core.feature_extractors.bev_vgg_pyramid import BevVggPyr

from odes.core.feature_extractors.img_vgg import ImgVgg
from odes.core.feature_extractors.img_vgg_pyramid import ImgVggPyr

from odes.core.feature_extractors.resnetv1 import resnetv1
from odes.core.feature_extractors.resnetv1_fpn import resnetv1_fpn


def get_extractor(extractor_config):

    extractor_type = extractor_config.WhichOneof('feature_extractor')

    # BEV feature extractors
    if extractor_type == 'bev_vgg':
        return BevVgg(extractor_config.bev_vgg)
    elif extractor_type == 'bev_vgg_pyr':
        return BevVggPyr(extractor_config.bev_vgg_pyr)

    # Image feature extractors
    elif extractor_type == 'img_vgg':
        return ImgVgg(extractor_config.img_vgg)
    elif extractor_type == 'img_vgg_pyr':
        return ImgVggPyr(extractor_config.img_vgg_pyr)
    elif extractor_type == 'resnetv1':
        return resnetv1(extractor_config.resnetv1 )  ## change it into resnet configuration later
    elif extractor_type == 'resnetv1_fpn':
        return resnetv1_fpn(extractor_config.resnetv1_fpn )

    return ValueError('Invalid feature extractor type', extractor_type)
