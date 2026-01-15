# FloodTransformer

A deep learning project for flood prediction using Transformer architecture. This project implements a FloodTransformer model that predicts water levels and inundation status using historical water level data, rainfall data, and DEM information.


## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended for training)
- Conda or pip package manager

### Install Dependencies

1. Clone the repository:
```bash
git clone https://github.com/jiachenkang/FloodTransformer.git
cd FloodTransformer
```

2. Create a conda environment (recommended):
```bash
conda create -n flood python=3.11
conda activate flood
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Python Packages
- torch==2.5.1
- netcdf4==1.7.2
- numpy==2.2.2
- matplotlib==3.10.0
- wandb==0.19
- pathlib

## Data Requirements

The model expects the following data files in the `data/` directory:

### Training/Validation Data
- NetCDF files (`.nc`) containing:
  - `Mesh2D_s1`: Water level time series
  - `Mesh2D_rain`: Rainfall data
  - `Mesh2D_ucx`, `Mesh2D_ucy`: Flow velocity in the X and Y direction
  - `Mesh2DContour_x`, `Mesh2DContour_y`: Grid coordinates


## Testing

### Running Training
To train the FloodTransformer model:

```bash
cd src
python train.py
```


## Model Inference
After training, you can use the saved model for inference:

1. Load the trained model:
```python
import torch
from model import FloodTransformer

# Load model
model = FloodTransformer()

# Load trained weights
checkpoint = torch.load('checkpoints/{timestamp}/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```


2. Run inference:
```python
with torch.no_grad():
    water_level_pred, has_water_pred = model(
        water_level_data, 
        rain_data, 
        dem_embed, 
        side_lens, 
        square_centers
    )
```



## License

Our code is released under the MIT License.

## Citation

Please consider citing our work, if you find this repository useful:

```
@article{gu5374815floodtransformer,
    title={Floodtransformer: Efficient Real-Time High-Resolution Flood Forecasting},
    author={GU, ZHANZHONG and Kang, Jiachen and Jin, Wenzheng and Tong, Feifei and Guo, Y Jay and Jia, Wenjing},
    journal={Available at SSRN 5374815}
}
```

