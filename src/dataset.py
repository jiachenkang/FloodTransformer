import netCDF4 as nc
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    def __init__(self, data_dir, config):
        self.config = config
        self.data_dir = Path(data_dir)
        self.nc_files = list(self.data_dir.glob('*.nc'))
        
        self.interval = config['start_time_interval']  
        self.pred_length = config['pred_length']  
        
        # create file_index_map, store (i, file, start_step) information
        self.file_index_map = {}
        self.total_samples = 0
        
        # through each nc file, determine valid starting time steps
        for file_idx, nc_file in enumerate(self.nc_files):
            with nc.Dataset(nc_file, 'r') as dataset:
                # get total time steps
                total_steps = min(config['end_step'], dataset.variables['Mesh2D_s1'].shape[0])
                
                # get possible start steps in this file
                possible_start_steps = []
                current_step = 0
                
                while current_step + self.pred_length < total_steps:
                    possible_start_steps.append(current_step)
                    current_step += self.interval
                
                # create index mapping for each valid starting time step
                for start_step in possible_start_steps:
                    self.file_index_map[self.total_samples] = (file_idx, start_step)
                    self.total_samples += 1

        self.water_level_min = torch.load('data/water_level_min.pt', weights_only=True).numpy()
        self.dem_min = torch.load('data/dem_min_tensor.pt', weights_only=True).numpy()
        self.water_level_scale = config['water_level_scale']
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx, start_step = self.file_index_map[idx]
        nc_file = self.nc_files[file_idx]
        end_step = start_step + self.pred_length

        with nc.Dataset(nc_file, 'r') as dataset:
            water_level = dataset.variables['Mesh2D_s1'][start_step][:-12] # 47791
            target = dataset.variables['Mesh2D_s1'][start_step:end_step+1,:-12] # 49, 47791
            rain = dataset.variables['Mesh2D_rain'][start_step+1:end_step+1,:-12] # 48, 47791
            u = dataset.variables['Mesh2D_ucx'][start_step+1:end_step+1,:-12] # 48, 47791
            v = dataset.variables['Mesh2D_ucy'][start_step+1:end_step+1,:-12] # 48, 47791
        
        # mask water_level and target below self.dem_min
        water_level = np.ma.masked_where(water_level < self.dem_min, water_level)
        target = np.ma.masked_where(target < self.dem_min, target)

        # standardization
        water_level = (water_level - self.water_level_min) / self.water_level_scale
        target = (target - self.water_level_min) / self.water_level_scale

        # create has_water arrays
        has_water = (~water_level.mask).astype(int) # 47791
        has_water_target = (~target.mask).astype(int) # 49, 47791

        # replace masked part of water_level with 0 and remove mask
        water_level = water_level.filled(0)
        target = target.filled(0)

        target_diff = target[1:] - target[:-1] # 48, 47791

        # replace nan in rain with 0
        rain[rain < 0] = 0
        rain = np.nan_to_num(rain, 0)

        # convert to tensor
        water_level = torch.tensor(water_level).unsqueeze(-1)
        rain = torch.tensor(rain).transpose(0,1)  # 47791, 48
        u_target = torch.tensor(u).transpose(0,1) # 47791, 48
        v_target = torch.tensor(v).transpose(0,1) # 47791, 48
        water_level_target = torch.tensor(target_diff).transpose(0,1) # 47791, 48
        has_water = torch.tensor(has_water).unsqueeze(-1)
        has_water_target = torch.tensor(has_water_target[1:]).transpose(0,1) # 47791, 48

        # concat water_level, has_water to (47791, 2)
        data = torch.cat((water_level, has_water), dim=1)

        return data, rain, u_target, v_target, water_level_target, has_water_target
