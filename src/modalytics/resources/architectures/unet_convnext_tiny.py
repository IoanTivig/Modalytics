"""U-Net (ConvNeXt-Tiny): U-Net with a modern convolutional encoder (2022 design).

A built-in segmentation architecture (Settings › Architectures): a decoder and an
encoder of segmentation-models-pytorch. See modalytics/training/architectures.py for
the format of this file.
"""

#: properties (edited on the Properties tab; written here on save)
KEY = 'unet_convnext_tiny'
NAME = 'U-Net (ConvNeXt-Tiny)'
DESCRIPTION = 'U-Net with a modern convolutional encoder (2022 design).'
DECODER = 'Unet'
ENCODER = 'tu-convnext_tiny'
DECODER_OPTIONS = {}
RECOMMENDED = True
REFERENCES = [
    {'text': 'Ronneberger O, Fischer P, Brox T (2015). U-Net: Convolutional Networks for Biomedical Image '
             'Segmentation. MICCAI 2015, LNCS 9351: 234–241.',
     'doi': '10.1007/978-3-319-24574-4_28'},
    {'text': 'Liu Z, Mao H, Wu CY, Feichtenhofer C, Darrell T, Xie S (2022). A ConvNet for the 2020s. CVPR 2022: '
             '11966–11976.',
     'doi': '10.1109/CVPR52688.2022.01167'},
]
