# nnU-Net customization

This repository includes a customized version of the official nnU-Net v2 source code under `model/nnUnet-master`.

The customized nnU-Net code is based on the official nnU-Net repository and was modified only for integration with the proposed preprocessing and inference workflow.

The following files were modified:

- `nnunetv2/inference/data_iterators.py`
- `nnunetv2/inference/predict_from_raw_data.py`
- `nnunetv2/preprocessing/preprocessors/default_preprocessor.py`

The core nnU-Net architecture and self-configuring training pipeline remain unchanged.