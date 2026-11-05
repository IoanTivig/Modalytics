"""U-Net++ (EfficientNet-B3): Nested, dense skip connections; a compound-scaled encoder (the shipped gland models).

A built-in segmentation architecture (Settings › Architectures): a decoder and an
encoder of segmentation-models-pytorch. See modalytics/training/architectures.py for
the format of this file.
"""

#: properties (edited on the Properties tab; written here on save)
KEY = 'unetpp_efficientnet_b3'
NAME = 'U-Net++ (EfficientNet-B3)'
DESCRIPTION = 'Nested, dense skip connections; a compound-scaled encoder (the shipped gland models).'
DECODER = 'UnetPlusPlus'
ENCODER = 'efficientnet-b3'
DECODER_OPTIONS = {}
RECOMMENDED = False
REFERENCES = [
    {'text': 'Zhou Z, Rahman Siddiquee MM, Tajbakhsh N, Liang J (2018). UNet++: A Nested U-Net Architecture for '
             'Medical Image Segmentation. DLMIA 2018, LNCS 11045: 3–11.',
     'doi': '10.1007/978-3-030-00889-5_1'},
    {'text': 'Tan M, Le QV (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. '
             'ICML 2019, PMLR 97: 6105–6114. arXiv:1905.11946'},
]
