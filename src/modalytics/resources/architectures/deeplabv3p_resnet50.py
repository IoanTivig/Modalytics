"""DeepLabV3+ (ResNet-50): Atrous (dilated) convolutions see wide context at once, with a light decoder for borders.

A built-in segmentation architecture (Settings › Architectures): a decoder and an
encoder of segmentation-models-pytorch. See modalytics/training/architectures.py for
the format of this file.
"""

#: properties (edited on the Properties tab; written here on save)
KEY = 'deeplabv3p_resnet50'
NAME = 'DeepLabV3+ (ResNet-50)'
DESCRIPTION = 'Atrous (dilated) convolutions see wide context at once, with a light decoder for borders.'
DECODER = 'DeepLabV3Plus'
ENCODER = 'resnet50'
DECODER_OPTIONS = {}
RECOMMENDED = True
REFERENCES = [
    {'text': 'Chen LC, Zhu Y, Papandreou G, Schroff F, Adam H (2018). Encoder-Decoder with Atrous Separable '
             'Convolution for Semantic Image Segmentation. ECCV 2018, LNCS 11211: 833–851.',
     'doi': '10.1007/978-3-030-01234-2_49'},
    {'text': 'He K, Zhang X, Ren S, Sun J (2016). Deep Residual Learning for Image Recognition. CVPR 2016: '
             '770–778.',
     'doi': '10.1109/CVPR.2016.90'},
]
