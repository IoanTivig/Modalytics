"""U-Net (ResNet-18): The classic encoder-decoder with skip connections; a light residual encoder.

A built-in segmentation architecture (Settings › Architectures): a decoder and an
encoder of segmentation-models-pytorch. See modalytics/training/architectures.py for
the format of this file.
"""

#: properties (edited on the Properties tab; written here on save)
KEY = 'unet_resnet18'
NAME = 'U-Net (ResNet-18)'
DESCRIPTION = 'The classic encoder-decoder with skip connections; a light residual encoder.'
DECODER = 'Unet'
ENCODER = 'resnet18'
DECODER_OPTIONS = {}
RECOMMENDED = False
REFERENCES = [
    {'text': 'Ronneberger O, Fischer P, Brox T (2015). U-Net: Convolutional Networks for Biomedical Image '
             'Segmentation. MICCAI 2015, LNCS 9351: 234–241.',
     'doi': '10.1007/978-3-319-24574-4_28'},
    {'text': 'He K, Zhang X, Ren S, Sun J (2016). Deep Residual Learning for Image Recognition. CVPR 2016: '
             '770–778.',
     'doi': '10.1109/CVPR.2016.90'},
]
