
# ST-ResNet for CSI Prediction

This project implements a spatial-temporal residual network (ST-ResNet) to predict symbol-level channel state information (CSI) using Rician fading channels with spatial and temporal correlation.

## Files

- `generate_csi.py`: Generates synthetic CSI data with Rician fading.
- `st_resnet.py`: Defines the ST-ResNet architecture.
- `train.py`: Training loop using data augmentation over channel correlation parameters.

## Usage

```bash
python train.py
```
