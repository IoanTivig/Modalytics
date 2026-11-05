"""U-Net (EfficientNet-B0): U-Net with the smallest EfficientNet: small and light.

A built-in segmentation architecture (Settings › Architectures): a decoder and an
encoder of segmentation-models-pytorch. See modalytics/training/architectures.py for
the format of this file.
"""

#: properties (edited on the Properties tab; written here on save)
KEY = 'unet_efficientnet_b0'
NAME = 'U-Net (EfficientNet-B0)'
DESCRIPTION = 'U-Net with the smallest EfficientNet: small and light.'
DECODER = 'Unet'
ENCODER = 'efficientnet-b0'
DECODER_OPTIONS = {}
RECOMMENDED = True
REFERENCES = [
    {'text': 'Ronneberger O, Fischer P, Brox T (2015). U-Net: Convolutional Networks for Biomedical Image '
             'Segmentation. MICCAI 2015, LNCS 9351: 234–241.',
     'doi': '10.1007/978-3-319-24574-4_28'},
    {'text': 'Tan M, Le QV (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. '
             'ICML 2019, PMLR 97: 6105–6114. arXiv:1905.11946'},
]
