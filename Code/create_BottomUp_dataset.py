import utils
import numpy as np
import xarray as xr
import pandas as pd

'''
 create_BottomUp_dataset.py

 This program combines all the dataset to produce a regional CO2 budget averaged over 2015-20.
 This dataset is written to: 

 This program takes as input the state/regional totals that were calculated in:
      - aggregate_tabular_carbon_datasets.py
      - Aggregate_TopDown_regional.py
      - aggregate_agricultural_inventory.py


'''

# ====

def split_to_regions(CONUS_flux):
    lat, lon, Regions, area = utils.Regional_mask('005')

    # Calculate fractional area of CONUS for each region
    regional_area = [np.sum(area[Regions == i+1]) for i in range(7)]
    regional_area_frac = np.array(regional_area) / np.sum(regional_area)

    # Distribute CONUS totals to regions
    regional_flux = {key: value * regional_area_frac for key, value in CONUS_flux.items()}
    
    return pd.DataFrame(regional_flux)
# -------------------------------


def estimate_regional_fluxes_from_CONUS_totals(Crop_inv, Forest_inv):
    """
    Estimate regional fluxes from CONUS totals for forest, crop, and water inventories.

    Parameters:
    Crop_inv (pd.DataFrame): Crop inventory data.
    Forest_inv (pd.DataFrame): Forest inventory data.

    Returns:
    tuple: Updated Crop_inv, Forest_inv, and River_inv dataframes.
    """
    
    # Define fluxes that were only available as CONUS totals
    CONUS_flux = {
        'aquatic burial': -20.6,  # TgC/yr
        'Coastal carbon export': -41.5  # TgC/yr
    }

    # -------- FORESTS --------
    PIC_SWDS_total = -14.0 - 18.6
    net_wood_trade = 16.5 - 19.7 # import - export
    # Assume that carbon uptake ending up in PIC, SWDS, and exported has the same spatial structure as forest harvest, such that:
    # Forest Harvest = Forest Harvest_adj + PIC and SWDS + trade
    Forest_inv['Harvest (adjusted)'] = Forest_inv['Harvest'] * ((Forest_inv['Harvest'].sum() - PIC_SWDS_total - net_wood_trade)/Forest_inv['Harvest'].sum())
    Forest_inv['PIC and SWDS'] = Forest_inv['Harvest'] * ((PIC_SWDS_total)/Forest_inv['Harvest'].sum())
    Forest_inv['trade'] = Forest_inv['Harvest'] * ((net_wood_trade)/Forest_inv['Harvest'].sum())

    # Calculate residual wood flux for forests to balance lateral fluxes
    C_in = np.abs(Forest_inv['Harvest (adjusted)'].sum())
    C_out = Forest_inv['Biofuel'].sum()
    CONUS_flux['residual wood'] = C_in - C_out
    # -------------------------

    # -------- CROPS --------
    # landfill stockchange
    crop_landfill_stockchange = -0.9
    net_crop_trade = 15.4 - 65.4 # import - export
    # Assume that uptake ending up in landfills or exported have the same spatial structure as crop harvest, such that:
    # Crop Harvest = Crop Harvest_adj + landfills + trade
    Crop_inv['Harvest (adjusted)'] = Crop_inv['Harvest'] * ((Crop_inv['Harvest'].sum() - crop_landfill_stockchange - net_crop_trade)/Crop_inv['Harvest'].sum())
    Crop_inv['landfill crops'] = Crop_inv['Harvest'] * ((crop_landfill_stockchange)/Crop_inv['Harvest'].sum())
    Crop_inv['trade'] = Crop_inv['Harvest'] * ((net_crop_trade)/Crop_inv['Harvest'].sum())
    # Calculate residual crop flux to balance lateral fluxes
    C_in = np.abs(Crop_inv['Harvest (adjusted)'].sum())
    C_out = Crop_inv['Biofuel'].sum() + Crop_inv['Livestock Respiration'].sum() + Crop_inv['Human Respiration'].sum()
    CONUS_flux['residual crop'] = C_in - C_out
    # -----------------------

    # Split CONUS totals to regions
    regional_estimate = split_to_regions(CONUS_flux)
    regional_estimate.index = Crop_inv.index


    # Add flux estimates to Forest inventory
    Forest_inv.loc[:, 'residual'] = regional_estimate['residual wood']

    # Add flux estimates to Crop inventory
    Crop_inv.loc[:, 'residual'] = regional_estimate['residual crop']

    # Create River inventory from relevant flux estimates
    River_inv = regional_estimate[['aquatic burial', 
                                   'Coastal carbon export']]

    # Remove total harvest field (since we have adjusted harvests now)
    Crop_inv = Crop_inv.drop(columns=['Harvest'])
    Forest_inv = Forest_inv.drop(columns=['Harvest'])

    # add index term
    Forest_inv.index.name = 'Region'
    Crop_inv.index.name = 'Region'
    River_inv.index.name = 'Region'

    return Crop_inv, Forest_inv, River_inv
# ------------------------------------------------------------------


def split_CropGrass(Regional_fluxes, Regional_Human_Resp, Regional_crop_grass_stock):
    """
    Splits and processes regional flux data into CropGrass and Forest inventories.

    Parameters:
    Regional_fluxes (pd.DataFrame): DataFrame containing various regional fluxes including biofuel, crop yield, etc.
    Regional_Human_Resp (pd.DataFrame): DataFrame containing human respiration data.
    Regional_crop_grass_stock (pd.DataFrame): DataFrame containing crop and grass stock data.

    Returns:
    tuple: Two DataFrames, CropGrass_inventory and Forest_inventory.
           CropGrass_inventory contains combined and renamed columns for crop and grass-related data.
           Forest_inventory contains selected and renamed columns for forest-related data.
    """
    # Select relevant columns for CropGrass inventory and create a copy to avoid SettingWithCopyWarning
    crop_grass_columns = ['Biofuel crop', 'Incineration', 'Crop yield', 'Livestock Respiration']
    Regional_fluxes_selected = Regional_fluxes[crop_grass_columns].copy()

    # Combine 'Biofuel crop' and 'Incineration' into a new column 'Biofuel'
    Regional_fluxes_selected['Biofuel'] = Regional_fluxes_selected['Biofuel crop'] + Regional_fluxes_selected['Incineration']
    
    # Drop the original 'Biofuel crop' and 'Incineration' columns
    Regional_fluxes_selected.drop(columns=['Biofuel crop', 'Incineration'], inplace=True)
    
    # Concatenate the selected columns with human respiration and crop grass stock data
    CropGrass_inventory = pd.concat([Regional_fluxes_selected, Regional_Human_Resp, Regional_crop_grass_stock], axis=1)
    
    # Rename columns for consistency
    CropGrass_inventory.rename(columns={'Crop yield': 'Harvest'}, inplace=True)

    # Select relevant columns for Forest inventory and create a copy to avoid SettingWithCopyWarning
    forest_columns = ['Biofuel wood', 'Forest Harvest', 'DeltaC_forest']
    Forest_inventory = Regional_fluxes[forest_columns].copy()
    
    # Rename columns for consistency
    Forest_inventory.rename(columns={
        'Biofuel wood': 'Biofuel',
        'Forest Harvest': 'Harvest',
    }, inplace=True)

    return CropGrass_inventory, Forest_inventory
# ----------------------------------------------------------------------------------


def read_datasets_and_calculate_regions(states,regions):
    
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

    return Regional_fluxes, Regional_Human_Resp, Regional_crop_grass_stock
# ------------------------------------------------------


def calculate_summary_dataset(Regional_fluxes,CropGrass_inventory, Forest_inventory, River_inventory):
    
    # Calculating combined variables
    combined_biofuel = Forest_inventory['Biofuel'] + CropGrass_inventory['Biofuel']
    combined_respiration = CropGrass_inventory['Human Respiration'] + CropGrass_inventory['Livestock Respiration']
    combined_trade = Forest_inventory['trade'] + CropGrass_inventory['trade']
    combined_residual = Forest_inventory['residual'] + CropGrass_inventory['residual']
    combined_stockchange = (Forest_inventory['DeltaC_forest'] + 
                            Forest_inventory['PIC and SWDS'] + 
                            CropGrass_inventory['DeltaC_grassland'] + 
                            CropGrass_inventory['DeltaC_cropland'] + 
                            CropGrass_inventory['landfill crops'] +
                            River_inventory['aquatic burial'])
    
    # Creating the new DataFrame
    summary_df = pd.DataFrame({
        'FF and IPPU': Regional_fluxes['FF and IPPU'],
        'DeltaC_total': combined_stockchange,
        'Crop Harvest (adjusted)': CropGrass_inventory['Harvest (adjusted)'],
        'Forest Harvest (adjusted)': Forest_inventory['Harvest (adjusted)'],
        'Respiration': combined_respiration,
        'Biofuel': combined_biofuel,
        'Trade': combined_trade,
        'Residual': combined_residual,
        'Coastal carbon export': River_inventory['Coastal carbon export']
    })

    summary_df.index.name = 'Region'

    return summary_df
# -----------------------------------------------------------------------------------


def main():
    # Read state & regional info
    states = utils.create_state_dataframe()
    regions = utils.define_state_groupings()

    # Read tabular state data and calculate regional totals
    # Read annual human respiration and calculate 2015-20 mean
    # Read annual cropland/grassland stockchange regional totals and calculate 2015-20 mean
    Regional_fluxes, Regional_Human_Resp, Regional_crop_grass_stock = read_datasets_and_calculate_regions(states,regions)
    
    # Split into Forest and Crop/Grass inventories
    CropGrass_inventory, Forest_inventory = split_CropGrass(Regional_fluxes, Regional_Human_Resp, Regional_crop_grass_stock)

    # Fluxes only available as CONUS totals (area weighted to regional)
    CropGrass_inventory, Forest_inventory, River_inventory = estimate_regional_fluxes_from_CONUS_totals(CropGrass_inventory,Forest_inventory)

    # Save datasets
    Forest_inventory.to_csv('../Data_processed/Regional_CO2_Budget_Forests.csv')
    CropGrass_inventory.to_csv('../Data_processed/Regional_CO2_Budget_CropGrass.csv')
    River_inventory.to_csv('../Data_processed/Regional_CO2_Budget_River.csv')

    # Create dataset that has FF+IPPU, total stock change, and aggregate lateral fluxes
    summary = calculate_summary_dataset(Regional_fluxes,CropGrass_inventory, Forest_inventory, River_inventory)

    # Save summary dataset
    summary.to_csv('../Data_processed/Regional_CO2_Budget_Summary.csv')


if __name__ == '__main__':
    main()
