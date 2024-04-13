import utils
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
        
    # Read cropland and grassland regional totals and calculate 2015-20 mean
    Regional_crop_grass_stock_eachYear = xr.open_dataset('../Data_processed/Regional_agricultural_stockchange.nc')
    Regional_crop_grass_stock = Regional_crop_grass_stock_eachYear.mean(dim='year').to_dataframe()
    
    ## Read OCO2 MIP regional total, calculate 2015-20 mean and standard deviations
    #Regional_TopDown_eachYear = xr.open_dataset('../Data_processed/Regional_OCO2_MIP.nc')
    #Regional_TopDown = Regional_TopDown_eachYear.mean(dim='year')
    #Regional_TopDown_mean = Regional_TopDown.mean(dim='model').to_dataframe()
    #Regional_TopDown_mean.columns = [f'{col} mean' if col != 'region' else col for col in Regional_TopDown_mean.columns]
    #Regional_TopDown_std = Regional_TopDown.std(dim='model').to_dataframe()
    #Regional_TopDown_std.columns = [f'{col} std' if col != 'region' else col for col in Regional_TopDown_std.columns]

    # Read OCO2 MIP regional total, calculate 2015-20 mean and standard deviations
    Regional_TopDown = xr.open_dataset('../Data_processed/Regional_OCO2_MIP_2015to2020avg.nc')

    Regional_TopDown_median = Regional_TopDown.sel(stat='median').to_dataframe()
    Regional_TopDown_median.drop(columns=['stat'], inplace=True)
    Regional_TopDown_median.columns = [f'{col} median' if col != 'region' else col for col in Regional_TopDown_median.columns] 

    Regional_TopDown_std = Regional_TopDown.sel(stat='std').to_dataframe()
    Regional_TopDown_std.drop(columns=['stat'], inplace=True)
    Regional_TopDown_std.columns = [f'{col} std' if col != 'region' else col for col in Regional_TopDown_std.columns] 


    # Combine all data into a Regional 2015-20 mean dataset
    Regional_CO2_Budget = pd.concat([Regional_fluxes, Regional_crop_grass_stock, Regional_TopDown_median, Regional_TopDown_std], axis=1)
    Regional_CO2_Budget.index.name = 'Region'
    Regional_CO2_Budget.to_csv('../Data_processed/Regional_CO2_Budget.csv')
    
