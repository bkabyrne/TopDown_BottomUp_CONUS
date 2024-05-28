import utils
import numpy as np
import xarray as xr
import pandas as pd

'''
 combined_topdown_and_bottomup.py

 This program combines all the dataset to produce a regional CO2 budget averaged over 2015-20.
 This dataset is written to: 

 This program takes as input the state/regional totals that were calculated in:
      - aggregate_tabular_carbon_datasets.py
      - Aggregate_TopDown_regional.py
      - aggregate_agricultural_inventory.py


'''

# ====                                                       

def estimate_regional_fluxes_from_CONUS_totals():

    # Fluxes that were only available as CONUS totals
    CONUS_flux = {}
    # Forest                                                        
    CONUS_flux['residual wood'] = 67.7 # TgC/yr         
    CONUS_flux['PIC and SWDS stockchange'] = -14.0 - 18.6 # TgC/yr   
    CONUS_flux['wood trade'] = 16.5 - 19.7 # TgC/yr     
    # Crop                                                          
    CONUS_flux['residual crop'] = 28.3 # TgC/yr !!!!! ADJUST WITH NEW HUMAN RESPIRATION NUMBERS !!!!!!!
    CONUS_flux['crop landfill stockchange'] = -0.9 # TgC/yr           
    CONUS_flux['crop trade'] = 15.1 - 65.4 # TgC/yr     
    # Riverine                                                      
    CONUS_flux['Lake and River emissions'] = 69.3 + 16.0 # TgC/yr
    CONUS_flux['Lake and River carbon burial'] = -20.6 # TgC/yr     
    CONUS_flux['Coastal carbon export'] = -41.5 # TgC/yr
    
    lat, lon, Regions, area = utils.Regional_mask('005')

    # Calculate fractional area of CONUS for each region     
    Regional_area = []
    for i in range(7):
        Regional_area.append(np.sum(area[np.where(Regions == i+1)]))
    Regional_area_frac = Regional_area / sum(Regional_area)

    # Distribute CONUS totals to regions                     
    regional_estimate_temp ={}
    for key, value in CONUS_flux.items():
        regional_estimate_temp[key] = value * Regional_area_frac
    regional_estimate = pd.DataFrame(regional_estimate_temp)

    return regional_estimate

# ====


if __name__ == '__main__':
    
    # Read state & regional info
    states = utils.create_state_dataframe()
    regions = utils.define_state_groupings()
    
    # Read tabular state totals and calculate state totals
    state_fluxes = pd.read_csv("../Data_processed/tabulated_fluxes_mean.csv")
    Regional_fluxes_dict= {}
    for region in regions:
        Regional_fluxes_dict[region] = state_fluxes[state_fluxes['State name'].isin(regions[region])].iloc[:,3:].sum(axis=0)
    Regional_fluxes = pd.DataFrame.from_dict(Regional_fluxes_dict, orient='index')

    # Read Human respiration regional totals and calculate 2015-19 mean
    Regional_Human_Resp_eachYear = xr.open_dataset('../Data_processed/Human_respiration_regional.nc')
    Regional_Human_Resp = Regional_Human_Resp_eachYear.mean(dim='year').to_dataframe()

    # Read cropland and grassland regional totals and calculate 2015-20 mean
    Regional_crop_grass_stock_eachYear = xr.open_dataset('../Data_processed/Regional_agricultural_stockchange.nc')
    Regional_crop_grass_stock = Regional_crop_grass_stock_eachYear.mean(dim='year').to_dataframe()
    
    # Read OCO2 MIP regional total, calculate 2015-20 mean and standard deviations
    Regional_TopDown = xr.open_dataset('../Data_processed/Regional_OCO2_MIP_2015to2020avg.nc')
    # ----------
    Regional_TopDown_median = Regional_TopDown.sel(stat='median').to_dataframe()
    Regional_TopDown_median.drop(columns=['stat'], inplace=True)
    Regional_TopDown_median.columns = [f'{col} median' if col != 'region' else col for col in Regional_TopDown_median.columns] 
    # ----------
    Regional_TopDown_std = Regional_TopDown.sel(stat='std').to_dataframe()
    Regional_TopDown_std.drop(columns=['stat'], inplace=True)
    Regional_TopDown_std.columns = [f'{col} std' if col != 'region' else col for col in Regional_TopDown_std.columns] 

    # Fluxes only available as CONUS totals (area weighted to regional)
    regional_estimate = estimate_regional_fluxes_from_CONUS_totals()
    regional_estimate.index = Regional_crop_grass_stock.index
    
    # Combine all data into a Regional 2015-20 mean dataset
    Regional_CO2_Budget = pd.concat([Regional_fluxes, Regional_Human_Resp, Regional_crop_grass_stock, regional_estimate, Regional_TopDown_median, Regional_TopDown_std], axis=1)
    Regional_CO2_Budget.index.name = 'Region'
    Regional_CO2_Budget.to_csv('../Data_processed/Regional_CO2_Budget.csv')
    
