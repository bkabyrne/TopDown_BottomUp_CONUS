# --- import modules ---   
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import utils
import xarray as xr


'''
  aggregate_agricultural_inventory.py

  This program aggregates the agricultural carbon stockchanges from a 0.5 x 0.625 grid to U.S. 
  climate assessment regions. 

'''

def calc_Regional_fluxes(flux,area_grid,Region_mask):

    Region_net_all = np.zeros((6,7))
    for region_i in range(7):
        Region_mask_weighted = Region_mask[region_i,:,:] * area_grid
        for year_i in range(6):
            Region_net_all[year_i,region_i] = np.sum(flux[year_i,:,:] * Region_mask_weighted) / np.sum(Region_mask_weighted)

    return Region_net_all
# ------------------------------

if __name__ == '__main__':
    
    '''
    ag_carbon_inventory_2015_2020.nc  
    
    title: Agricultural Cropland and Grassland Carbon from 2022 US Greenhouse Gas Inventory
    institution: Natural Resource Ecology Laboratory at Colorado State University
    history: 2024-02-01 17:13:24.802242
    comment: Annual carbon values for soil carbon, litter, live C and production for United States
             cropland and grassland, modeled and extended to include all area, provided on 
             0.625 x 0.5 degree grid for 2015-2020.

    variables(dimensions):
        - time(time)
        - lat(lat)
        - lon(lon)
        - dSOC(time, lat, lon)
              --- Annual change in soil organic carbon  [metric_tons]
        - dlitter_C(time, lat, lon)
              --- Annual change in total C in structural and metabolic surface and soil litter and standing dead.  [metric_tons]
        - dlive_C(time, lat, lon)
              --- Annual change in end-of-year total live biomass  [metric_tons]
        - production_C(time, lat, lon) 
              --- Annual aboveground and root C production  [metric_tons]

    groups: cropland and grassland, cropland, grassland
    '''

    # Read the grided agricultural stockchanges
    fname = '../Data_input/ag_carbon_inventory_2015_2020.nc'
    f=Dataset(fname,mode='r')
    grassland_dC = (f.groups['grassland']['dSOC'][:] +
                    f.groups['grassland']['dlitter_C'][:] +
                    f.groups['grassland']['dlive_C'][:]) * 1e-6 # (metric tons * 1e-6 = TgC)
    cropland_dC = (f.groups['cropland']['dSOC'][:] +
                   f.groups['cropland']['dlitter_C'][:] +
                   f.groups['cropland']['dlive_C'][:]) * 1e-6 # (metric tons * 1e-6 = TgC)
    
    # Read mask information
    lat, lon, Regions, area = utils.Regional_mask('05x0625')
    
    # Calculate regional totals
    grassland_dC_region = calc_Regional_fluxes(grassland_dC,area,Regions)
    cropland_dC_region = calc_Regional_fluxes(cropland_dC,area,Regions)
    
    # Create xarray and save
    da_a = xr.DataArray(grassland_dC_region, dims=('time','region'), attrs={'units': 'TgC/yr'})
    da_b = xr.DataArray(cropland_dC_region, dims=('time','region'), attrs={'units': 'TgC/yr'})
    reg_dataset = xr.Dataset({'grassland': da_a, 'cropland': da_b})
    region_coords = ['Northwest','N Great Plains','Midwest','Southwest','S Great Plains','Southeast','Northeast']
    reg_dataset['region'] = region_coords
    time_coords = [2015,2016,2017,2018,2019,2020]
    reg_dataset['time'] = time_coords
    reg_dataset.to_netcdf('../Data_processed/Regional_agricultural_stockchange.nc')
    
    
