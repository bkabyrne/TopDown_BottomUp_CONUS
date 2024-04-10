import csv
import numpy as np
from netCDF4 import Dataset

'''
 read_lateral_fluxes.py

 This program reads the lateral fluxes carbon fluxes and compiles them to state totals and writes to a file.
 
 The lateral fluxes are:
    - biofuelds (wood, ethanol, bidiesal)
    - Landfills
    - Incineration
    - Fossil Fuels (FF) and Industrial Processes (IPPU)
    - Crop harvests
    - Livesotck respiration
    - Human respiration
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

def read_Biofuel_data():

    # -------------------------------------------
    # Loops of Biofuel csv files and compiles
    # -------------------------------------------

    BioFuel_file_names = ['/Users/bbyrne/Downloads/State_Biomass_Wood_Electric.csv',
        '/Users/bbyrne/Downloads/State_Biomass_Wood_Industrial.csv',
        '/Users/bbyrne/Downloads/State_Biomass_Wood_Residential.csv',
        '/Users/bbyrne/Downloads/State_Biomass_Wood_Commercial.csv',
        '/Users/bbyrne/Downloads/State_Biofuel_Ethanol_Transportation.csv',
        '/Users/bbyrne/Downloads/State_Biofuel_Ethanol_Commercial.csv',
        '/Users/bbyrne/Downloads/State_Biofuel_Ethanol_Industrial.csv',
        '/Users/bbyrne/Downloads/State_Biofuel_Biodiesel_Transportation.csv']

    # --- Biofuel wood --- 
    BioFuel_all_wood = np.zeros((4,6,51))
    for i in range(4):
        BioFuel_all_wood[i,:,:] = read_EPA_data(BioFuel_file_names[i],State_names) * 12./44. # CO2 -> C
    BioFuel_all_wood_total = np.sum(BioFuel_all_wood,0)

    # --- Biofuel ethanol ---
    BioFuel_all_ethanol = np.zeros((3,6,51))
    for i in range(3):
        BioFuel_all_ethanol[i,:,:] = read_EPA_data(BioFuel_file_names[i],State_names) * 12./44. # CO2 -> C
    BioFuel_all_ethanol_total = np.sum(BioFuel_all_ethanol,0)

    # --- Biodiesal ---
    BioFuel_all_biodiesal= np.zeros((1,6,51))
    for i in range(1):
        BioFuel_all_biodiesal[i,:,:] = read_EPA_data(BioFuel_file_names[i],State_names) * 12./44. # CO2 -> C
    BioFuel_all_biodiesal_total = np.sum(BioFuel_all_biodiesal,0)

    return BioFuel_all_wood_total, BioFuel_all_ethanol_total, BioFuel_all_biodiesal_total

# ========

def read_Landfill_data():

    # -------------------------------------------
    # Reads Landfill data
    # -------------------------------------------

    Landfill_names = ['/Users/bbyrne/Downloads/State_MSW_landfill.csv',
        '/Users/bbyrne/Downloads/State_Industrial_landfill.csv']

    Landfill_all_data = np.zeros((2,6,51))
    for i in range(2):
        Landfill_all_data[i,:,:] = read_EPA_data(Landfill_names[i],State_names)
    #print('Landfills CH4 in CO2 equivalent')
    # Estimates use 1 CO2 ; 25 CH4 100-Year GWP value
    #      Carbon dioxide equivalent (CO2e or CO2eq or CO2-e) of a quantity of 
    #      gas is calculated from its GWP. For any gas, it is the mass of CO2 
    #      which would warm the earth as much as the mass of that gas.[31] Thus 
    #      it provides a common scale for measuring the climate effects of different 
    #      gases. It is calculated as GWP multiplied by mass of the other gas. 
    #      For example, if a gas has GWP of 100, two tonnes of the gas have CO2e of 
    #      200 tonnes, and 9 tonnes of the gas has CO2e of 900 tonnes. 
    # So, CO2e = CH4 * GWP
    #          = TgCH4 * GWP
    # So covert to TgC = X * gC/gCH4 * (1/GWP)
    #                  = X * 12./16. * (1./25)
    Landfill_all_data_C_as_CH4 = Landfill_all_data * (12./16. * (1./25))
    # Assume total C emissions are double (because emissions are roughly half CO2 and CH4)
    Landfill_all_data_C = Landfill_all_data_C_as_CH4 * 2.
    Landfill_all_data_C_all = np.sum(Landfill_all_data_C,0)
    return Landfill_all_data_C_all

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

    return State_yield_TgC

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

def Human_resp(State_names):

    # -------------------------------------------
    # Reads Human respiration data, provided
    # as county totals and are aggregated to state totals
    # -------------------------------------------

    Human_Respiration_2015t = np.zeros(4000)+1e9
    Human_Respiration_2016t = np.zeros(4000)+1e9
    Human_Respiration_2017t = np.zeros(4000)+1e9
    Human_Respiration_2018t = np.zeros(4000)+1e9
    Human_Respiration_2019t = np.zeros(4000)+1e9
    human_resp_index_of_statet = np.zeros(4000)+1e9


    file_path="/Users/bbyrne/Downloads/US_counties_2010_2019.csv"
    lne=0
    n=0
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if lne>0:
                if isinstance(row[13], str):
                    Human_Respiration_2015t[n] = float(row[7]) * 1e-12
                    Human_Respiration_2016t[n] = float(row[8]) * 1e-12
                    Human_Respiration_2017t[n] = float(row[9]) * 1e-12
                    Human_Respiration_2018t[n] = float(row[10]) * 1e-12
                    Human_Respiration_2019t[n] = float(row[11]) * 1e-12
                    state = row[13].upper()
                    human_resp_index_of_statet[n] = State_names.index(state)
                    n += 1
            lne=lne+1


    Human_Respiration_2015 = Human_Respiration_2015t[0:n]
    Human_Respiration_2016 = Human_Respiration_2016t[0:n]
    Human_Respiration_2017 = Human_Respiration_2017t[0:n]
    Human_Respiration_2018 = Human_Respiration_2018t[0:n]
    Human_Respiration_2019 = Human_Respiration_2019t[0:n]
    human_resp_index_of_state = human_resp_index_of_statet[0:n]

    state_Human_Respiration_2015 = np.zeros(51)
    state_Human_Respiration_2016 = np.zeros(51)
    state_Human_Respiration_2017 = np.zeros(51)
    state_Human_Respiration_2018 = np.zeros(51)
    state_Human_Respiration_2019 = np.zeros(51)
    for s in range(51):
        II = np.where(human_resp_index_of_state == s)
        state_Human_Respiration_2015[s] = np.sum(Human_Respiration_2015[II])
        state_Human_Respiration_2016[s] = np.sum(Human_Respiration_2016[II])
        state_Human_Respiration_2017[s] = np.sum(Human_Respiration_2017[II])
        state_Human_Respiration_2018[s] = np.sum(Human_Respiration_2018[II])
        state_Human_Respiration_2019[s] = np.sum(Human_Respiration_2019[II])
    state_Human_Respiration_allyear = np.zeros((6,51))
    state_Human_Respiration_allyear[0,:] = state_Human_Respiration_2015
    state_Human_Respiration_allyear[1,:] = state_Human_Respiration_2016
    state_Human_Respiration_allyear[2,:] = state_Human_Respiration_2017
    state_Human_Respiration_allyear[3,:] = state_Human_Respiration_2018
    state_Human_Respiration_allyear[4,:] = state_Human_Respiration_2019
    state_Human_Respiration_allyear[5,:] = state_Human_Respiration_2019 # Assume 2020 respiration same as 2019
    mean_year_Human_Resp = (state_Human_Respiration_2015+
                            state_Human_Respiration_2016+
                            state_Human_Respiration_2017+
                            state_Human_Respiration_2018+
                            state_Human_Respiration_2019)/5.
    return state_Human_Respiration_allyear

# ========

def write_data( file_out, State_names, FF_IPPU_data, State_yield_TgC_2015to2020, BioFuel_all_wood_total, BioFuel_all_ethanol_total, 
               BioFuel_all_biodiesal_total, Landfill_all_data_C_all, Incineration_data, state_Human_Respiration_allyear, State_livestock_TgC):
    
    # -------------------------------------------
    # Writes out all the lateral flux arrays that have
    # been combined into (year,state) arrays
    # -------------------------------------------

    string_lengthN = 25

    dataset = Dataset(file_out,'w')
    years1 = dataset.createDimension('year',6)
    states1 = dataset.createDimension('state',51)
    dataset.createDimension('string_length', string_lengthN)
    #
    states_inds = dataset.createVariable('state_name', 'S1', ('state', 'string_length'))
    # Write string data to the variable
    for i, s in enumerate(State_names):
        padded_string = s.ljust(string_lengthN)[:string_lengthN]
        states_inds[i, :] = list(padded_string.encode('utf-8'))
    states_inds.longname = 'State names'
    #
    FF_IPPUs = dataset.createVariable('FF_IPPU', np.float64, ('year','state'))
    FF_IPPUs[:,:] = FF_IPPU_data
    FF_IPPUs.longname = 'FF + IPPU excluding waste incineration. From EPA state GHG data'
    FF_IPPUs.units = 'TgC year-1'
    #
    Crop_Yields = dataset.createVariable('Crop_Yield', np.float64, ('year','state'))
    Crop_Yields[:,:] = State_yield_TgC_2015to2020
    Crop_Yields.longname = 'Yield of agricultural crops. Estimated by Gabriel Dias Ferreira'
    Crop_Yields.units = 'TgC year-1'
    #
    Biofuel_woods = dataset.createVariable('Biofuel_wood', np.float64, ('year','state'))
    Biofuel_woods[:,:] = BioFuel_all_wood_total
    Biofuel_woods.longname = 'Wood. From EPA state GHG data'
    Biofuel_woods.units = 'TgC year-1'
    #
    Biofuel_ethanols = dataset.createVariable('Biofuel_ethanol', np.float64, ('year','state'))
    Biofuel_ethanols[:,:] = BioFuel_all_ethanol_total
    Biofuel_ethanols.longname = 'ethanol. From EPA state GHG data'
    Biofuel_ethanols.units = 'TgC year-1'
    #
    Biofuel_biodiesals = dataset.createVariable('Biofuel_biodiesal', np.float64, ('year','state'))
    Biofuel_biodiesals[:,:] = BioFuel_all_biodiesal_total
    Biofuel_biodiesals.longname = 'biodiesal. From EPA state GHG data'
    Biofuel_biodiesals.units = 'TgC year-1'
    #
    Landfills = dataset.createVariable('Landfill', np.float64, ('year','state'))
    Landfills[:,:] = Landfill_all_data_C_all
    Landfills.longname = 'Landfill C estimated from Landfill CH4. From EPA state GHG data'
    Landfills.units = 'TgC year-1'
    #
    Incinerations = dataset.createVariable('Incineration', np.float64, ('year','state'))
    Incinerations[:,:] = Incineration_data
    Incinerations.longname = 'Incineration of Waste. From EPA state GHG data'
    Incinerations.units = 'TgC year-1'
    #
    Respiration_Humans = dataset.createVariable('Respiration_Human', np.float64, ('year','state'))
    Respiration_Humans[:,:] = state_Human_Respiration_allyear
    Respiration_Humans.longname = 'Human Respiration estimated by Yinon Bar-On'
    Respiration_Humans.units = 'TgC year-1'
    #
    Respiration_Livestocks = dataset.createVariable('Respiration_Livestock', np.float64, ('year','state'))
    Respiration_Livestocks[:,:] = State_livestock_TgC
    Respiration_Livestocks.longname = 'Livestock Respiration estimated by Gabriel Dias Ferreira'
    Respiration_Livestocks.units = 'TgC year-1'
    dataset.close()

# ========
# ========
# ========
    
if __name__ == '__main__':

    State_names = ['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 
                'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 
                'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 
                'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA',
                'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 
                'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
                'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING', 
                'DISTRICT OF COLUMBIA']

    # Read Biofuel data
    BioFuel_all_wood_total, BioFuel_all_ethanol_total, BioFuel_all_biodiesal_total = read_Biofuel_data()
    print('=== Biofuel - wood ===')
    print(np.mean(np.sum(BioFuel_all_wood_total,1),0))
    print('=== Biofuel - ethanol ===')
    print(np.mean(np.sum(BioFuel_all_ethanol_total,1),0))
    print('=== Biofuel - biodiesal ===')
    print(np.mean(np.sum(BioFuel_all_biodiesal_total,1),0))

    # Read Landfill data
    Landfill_all_data_C_all = read_Landfill_data()
    print('=== Landfill ===')
    print(np.mean(np.sum(Landfill_all_data_C_all,1),0))

    # Read Incineration data
    Incineration_names = '/Users/bbyrne/Downloads/State_Incineration_of_Waste.csv'
    Incineration_data = read_EPA_data(Incineration_names,State_names) * 12./44. # CO2 -> C
    mean_year_Incineration = np.mean(Incineration_data,0)
    print('=== Incineration ===')
    print(np.mean(np.sum(Incineration_data,1),0))

    # Read FF & IPPU data
    FF_IPPU_names = '/Users/bbyrne/Downloads/State_FF_and_IPPU.csv'
    FF_IPPU_data = read_EPA_data(FF_IPPU_names,State_names) * 12./44. # CO2 -> C
    print('=== FF + IPPU ===')
    print(np.mean(np.sum(FF_IPPU_data,1),0))

    # Crop harvests
    file_path = '/Users/bbyrne/Downloads/crop_yield_counties.csv'
    State_yield_TgC_2015to2020 = read_crop_yield_data(file_path, State_names)
    print('=== Crop yield ===')
    print(np.sum(np.mean(State_yield_TgC_2015to2020, axis=0)))
    
    # Livestock respiration
    file_path = '/Users/bbyrne/Downloads/python_codes/livestock_resp_tCCO2_results_county_20240216.csv'
    State_livestock_TgC = read_livestock_data(file_path, State_names)
    print('=== livestock Respiration ===')
    print(np.sum(np.mean(State_livestock_TgC, axis=0)))

    #Human Respiration
    state_Human_Respiration_allyear = Human_resp(State_names)
    print('=== Human Respiration ===')
    print(np.sum(np.mean(state_Human_Respiration_allyear, axis=0)))

    file_out = 'FF_IPPU_and_lateral_fluxes_test.nc'
    write_data(file_out, State_names, FF_IPPU_data, State_yield_TgC_2015to2020, BioFuel_all_wood_total, BioFuel_all_ethanol_total, 
                BioFuel_all_biodiesal_total, Landfill_all_data_C_all, Incineration_data, state_Human_Respiration_allyear, State_livestock_TgC)