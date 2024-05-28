import csv
import utils
import numpy as np
import pandas as pd
from netCDF4 import Dataset

'''
 aggregate_tabular_carbon_datasets.py

 This program reads tabular carbon fluxes and compiles them to state totals and writes to a file.
 
 The tabulated fluxes are:
    - biofuelds (wood, ethanol, bidiesal)
    - Incineration
    - Fossil Fuels (FF) and Industrial Processes (IPPU)
    - Crop harvests
    - Livesotck respiration
    - Human respiration
    - Forest harvests
    - Forest inventory
'''

# ========

def read_EPA_data(input_file_name, State_names):

    # -------------------------------------------
    # Reads EPA data given expected csv format
    # Returns array of dim (year,state) for 2015-20
    # -------------------------------------------

    vec_state_out = np.zeros(51)
    vec_all_year = np.zeros((6, 51))

    with open(input_file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for n, row in enumerate(reader, start=1):
            state = row[0]
            vec_statet = State_names.index(state)
            vec_state_out[n - 1] = vec_statet
            vec_FLUXt = np.array([float(x) for x in row[2:8]])
            vec_all_year[:, vec_statet] += vec_FLUXt

    return vec_all_year

# ========

def read_Biofuel_data(State_names):

    # -------------------------------------------
    # Loops of Biofuel csv files and compiles
    # -------------------------------------------

    BioFuel_wood_file_names = ['../Data_input/State_Biomass_Wood_Electric.csv',
                          '../Data_input/State_Biomass_Wood_Industrial.csv',
                          '../Data_input/State_Biomass_Wood_Residential.csv',
                          '../Data_input/State_Biomass_Wood_Commercial.csv']
    
    BioFuel_ethanol_file_names = ['../Data_input/State_Biofuel_Ethanol_Transportation.csv',
                          '../Data_input/State_Biofuel_Ethanol_Commercial.csv',
                          '../Data_input/State_Biofuel_Ethanol_Industrial.csv']
    
    BioFuel_biodiesal_file_names = ['../Data_input/State_Biofuel_Biodiesel_Transportation.csv']
    
    # --- Biofuel wood --- 
    BioFuel_all_wood = np.zeros((4,6,51))
    for i in range(4):
        BioFuel_all_wood[i,:,:] = read_EPA_data(BioFuel_wood_file_names[i],State_names) * 12./44. # CO2 -> C
    BioFuel_all_wood_total = np.sum(BioFuel_all_wood,0)

    # --- Biofuel ethanol ---
    BioFuel_all_ethanol = np.zeros((3,6,51))
    for i in range(3):
        BioFuel_all_ethanol[i,:,:] = read_EPA_data(BioFuel_ethanol_file_names[i],State_names) * 12./44. # CO2 -> C
    BioFuel_all_ethanol_total = np.sum(BioFuel_all_ethanol,0)

    # --- Biodiesal ---
    BioFuel_all_biodiesal= np.zeros((1,6,51))
    for i in range(1):
        BioFuel_all_biodiesal[i,:,:] = read_EPA_data(BioFuel_biodiesal_file_names[i],State_names) * 12./44. # CO2 -> C
    BioFuel_all_biodiesal_total = np.sum(BioFuel_all_biodiesal,0)

    return BioFuel_all_wood_total, BioFuel_all_ethanol_total, BioFuel_all_biodiesal_total

# ========

def read_crop_yield_data(file_path, State_names):

    # -------------------------------------------
    # Reads crop yield data. These data are provided
    # as county totals and are aggregated to state totals
    # -------------------------------------------

    crop_year = []
    crop_index_of_state = []
    crop_county_yield = []

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for n, row in enumerate(reader, start=1):
            crop_year.append(int(row[3]))
            crop_index_of_state.append(State_names.index(row[0]))
            crop_county_yield.append(float(row[6]))

    crop_year = np.array(crop_year)
    crop_index_of_state = np.array(crop_index_of_state)
    crop_county_yield = np.array(crop_county_yield)

    State_yield = np.zeros((6, 51))
    for ii in range(6):
        II = np.where(crop_year == 2015 + ii)
        for jj in range(51):
            JJ = np.where(crop_index_of_state == jj)
            intersection = np.intersect1d(II, JJ)
            if intersection.size > 0:
                State_yield[ii, jj] += np.sum(crop_county_yield[intersection])
    # respiration [ton C of CO2] -> convert to TgC
    State_yield_TgC = State_yield * 1e-6

    return State_yield_TgC * (-1.) # Removed from atmosphere

# ========

def read_livestock_data(file_path, State_names):

    # -------------------------------------------
    # Reads livestrock respiration data, provided
    # as county totals and are aggregated to state totals
    # -------------------------------------------

    livestock_year = []
    livestock_index_of_state = []
    livestock_county_yield = []

    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for n, row in enumerate(reader, start=1):
            livestock_year.append(int(row[3]))
            livestock_index_of_state.append(State_names.index(row[0]))
            livestock_county_yield.append(float(row[4]))

    livestock_year = np.array(livestock_year)
    livestock_index_of_state = np.array(livestock_index_of_state)
    livestock_county_yield = np.array(livestock_county_yield)

    State_livestock = np.zeros((6, 51))
    for ii in range(6):
        II = np.where(livestock_year == 2015 + ii)
        for jj in range(51):
            JJ = np.where(livestock_index_of_state == jj)
            intersection = np.intersect1d(II, JJ)
            if intersection.size > 0:
                State_livestock[ii, jj] += np.sum(livestock_county_yield[intersection])
    # respiration [ton C of CO2] -> convert to TgC
    State_livestock_TgC = State_livestock * 1e-6

    return State_livestock_TgC

# ========

def read_forest_harvest_data(filename,State_codes):

    statecd = []
    removals = []

    with open(filename) as csvfile:
        next(csvfile)  # Skip header
        reader = csv.reader(csvfile)
        for row in reader:
            statecd.append(int(row[0]))
            removals.append(float(row[1]))

    state_removals = np.zeros(len(State_codes))
    for i, code in enumerate(State_codes):
        if code in statecd:
            state_removals[i] = removals[statecd.index(code)]

    return state_removals * (-1.) # Removed from atmosphere

# ========

def read_forest_inventory_data(filename,State_codes):
    # Load data into a DataFrame using pandas                                                               
    df = pd.read_csv(filename)
    # Calculate state totals using groupby and sum                                                          
    state_totals = df.groupby(['statecd', 'year']).sum()
    # Find years
    years = df['year'].unique()
    
    TotalCflux = np.zeros((6,51))
    for i, code in enumerate(State_codes):
        for j, year in enumerate(years):
            filtered_data = state_totals.loc[(state_totals.index.get_level_values('statecd') == code) & (state_totals.index.get_level_values('year') == year), 'Total C flux'].values
            if len(filtered_data) > 0:
                TotalCflux[j, i] = filtered_data[0]


    return TotalCflux


# ========

def create_dataframe( State_codes_in, State_abbrev_in, State_names_in, BioFuel_all_wood_total_in, BioFuel_all_ethanol_total_in,
                      BioFuel_all_biodiesal_total_in, Incineration_data_in, FF_IPPU_data_in,
                      State_yield_TgC_2015to2020_in, State_livestock_TgC_in, forest_harvest_removals_in,
                      Forest_inventory_in):
        
    # Create dataframe with all the data
    data_dict = {
        'State code': State_codes_in,
        'State abbr': State_abbrev_in,
        'State name': State_names_in,
        'Biofuel wood': BioFuel_all_wood_total_in,
        'Biofuel ethanol': BioFuel_all_ethanol_total_in,
        'Biofuel biodiesal': BioFuel_all_biodiesal_total_in,
        'Incineration': Incineration_data_in,
        'FF and IPPU': FF_IPPU_data_in,
        'Crop yield': State_yield_TgC_2015to2020_in,
        'Livestock Respiration': State_livestock_TgC_in,
        'Forest Harvest': forest_harvest_removals_in,
        'Forest inventory': Forest_inventory_in
        }
    
    df = pd.DataFrame(data_dict)
    
    return df

# ========
# ========
# ========
    
if __name__ == '__main__':

    # Read in US State info
    state_df = utils.create_state_dataframe()
    # convert info to lists
    State_codes = state_df['State Code'].tolist()
    State_abbrev = state_df['State Abbreviation'].tolist()
    State_names = state_df['State Name'].tolist()

    # Read Biofuel data
    BioFuel_all_wood_total, BioFuel_all_ethanol_total, BioFuel_all_biodiesal_total = read_Biofuel_data(State_names)
    print('=== Biofuel - wood ===')
    print(np.mean(np.sum(BioFuel_all_wood_total,1),0))
    print('=== Biofuel - ethanol ===')
    print(np.mean(np.sum(BioFuel_all_ethanol_total,1),0))
    print('=== Biofuel - biodiesal ===')
    print(np.mean(np.sum(BioFuel_all_biodiesal_total,1),0))

    # Read Incineration data
    Incineration_names = '../Data_input/State_Incineration_of_Waste.csv'
    Incineration_data = read_EPA_data(Incineration_names,State_names) * 12./44. # CO2 -> C
    print('=== Incineration ===')
    print(np.mean(np.sum(Incineration_data,1),0))

    # Read FF & IPPU data
    FF_IPPU_names = '../Data_input/State_FF_and_IPPU.csv'
    FF_IPPU_data = read_EPA_data(FF_IPPU_names,State_names) * 12./44. # CO2 -> C
    print('=== FF + IPPU ===')
    print(np.mean(np.sum(FF_IPPU_data,1),0))

    # Crop harvests
    file_path = '../Data_input/crop_yield_counties.csv'
    State_yield_TgC_2015to2020 = read_crop_yield_data(file_path, State_names)
    print('=== Crop yield ===')
    print(np.sum(np.mean(State_yield_TgC_2015to2020, axis=0)))
    
    # Livestock respiration
    file_path = '../Data_input/livestock_resp_tCCO2_results_county_20240216.csv'
    State_livestock_TgC = read_livestock_data(file_path, State_names)
    print('=== livestock Respiration ===')
    print(np.sum(np.mean(State_livestock_TgC, axis=0)))

    # Example usage:
    filename = '../Data_input/CMS_state_harvest_removals_082823.csv'
    forest_harvest_removals = read_forest_harvest_data(filename,State_codes)
    print('=== Forest Harvest ===')
    print(np.sum(forest_harvest_removals))

    filename = '../Data_input/county_level_FRF_and_LCF_flux_in_MMTC_2015-2020_20230711.csv'
    Forest_inventory = read_forest_inventory_data(filename,State_codes)
    print('=== Forest inventory ===')
    print(np.sum(np.mean(Forest_inventory, axis=0)))

    # Create dataframe and save
    for year in range(2015,2021):
        tabulated_fluxes = create_dataframe(State_codes,State_abbrev,State_names,BioFuel_all_wood_total[year-2015,:],BioFuel_all_ethanol_total[year-2015,:],
                                          BioFuel_all_biodiesal_total[year-2015,:],Incineration_data[year-2015,:],
                                          FF_IPPU_data[year-2015,:],State_yield_TgC_2015to2020[year-2015,:],State_livestock_TgC[year-2015,:],
                                          forest_harvest_removals,Forest_inventory[year-2015,:])
        tabulated_fluxes.to_csv('../Data_processed/tabulated_fluxes_'+str(year).zfill(4)+'.csv', index=False)


    tabulated_fluxes_mean = create_dataframe(State_codes,State_abbrev,State_names,np.mean(BioFuel_all_wood_total,0),np.mean(BioFuel_all_ethanol_total,0),
                                           np.mean(BioFuel_all_biodiesal_total,0),np.mean(Incineration_data,0),
                                           np.mean(FF_IPPU_data,0),np.mean(State_yield_TgC_2015to2020,0),np.mean(State_livestock_TgC,0),
                                           forest_harvest_removals,np.mean(Forest_inventory,0))
    tabulated_fluxes_mean.to_csv('../Data_processed/tabulated_fluxes_mean.csv', index=False)

