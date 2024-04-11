import pandas as pd
import numpy as np
from netCDF4 import Dataset
from math import pi, cos, radians

def create_state_dataframe():
    """
    Create a DataFrame from state data.

    Returns:
    pd.DataFrame: DataFrame containing state information.
    """

    state_data = """
    01|AL|Alabama|01779775
    02|AK|Alaska|01785533
    04|AZ|Arizona|01779777
    05|AR|Arkansas|00068085
    06|CA|California|01779778
    08|CO|Colorado|01779779
    09|CT|Connecticut|01779780
    10|DE|Delaware|01779781
    11|DC|District of Columbia|01702382
    12|FL|Florida|00294478
    13|GA|Georgia|01705317
    15|HI|Hawaii|01779782
    16|ID|Idaho|01779783
    17|IL|Illinois|01779784
    18|IN|Indiana|00448508
    19|IA|Iowa|01779785
    20|KS|Kansas|00481813
    21|KY|Kentucky|01779786
    22|LA|Louisiana|01629543
    23|ME|Maine|01779787
    24|MD|Maryland|01714934
    25|MA|Massachusetts|00606926
    26|MI|Michigan|01779789
    27|MN|Minnesota|00662849
    28|MS|Mississippi|01779790
    29|MO|Missouri|01779791
    30|MT|Montana|00767982
    31|NE|Nebraska|01779792
    32|NV|Nevada|01779793
    33|NH|New Hampshire|01779794
    34|NJ|New Jersey|01779795
    35|NM|New Mexico|00897535
    36|NY|New York|01779796
    37|NC|North Carolina|01027616
    38|ND|North Dakota|01779797
    39|OH|Ohio|01085497
    40|OK|Oklahoma|01102857
    41|OR|Oregon|01155107
    42|PA|Pennsylvania|01779798
    44|RI|Rhode Island|01219835
    45|SC|South Carolina|01779799
    46|SD|South Dakota|01785534
    47|TN|Tennessee|01325873
    48|TX|Texas|01779801
    49|UT|Utah|01455989
    50|VT|Vermont|01779802
    51|VA|Virginia|01779803
    53|WA|Washington|01779804
    54|WV|West Virginia|01779805
    55|WI|Wisconsin|01779806
    56|WY|Wyoming|01779807
    """
    
    
    # Split the data into lines and then split each line into fields
    state_lines = state_data.strip().split('\n')
    state_records = []

    # Create a list of dictionaries where each dictionary represents a state record
    for line in state_lines:
        fields = line.split('|')
        state_records.append({'State Code': int(fields[0]), 'State Abbreviation': fields[1], 'State Name': fields[2].upper(), 'STATENS': fields[3]})

    # Create a DataFrame from the list of dictionaries
    state_df = pd.DataFrame(state_records)
    return state_df

# ==================

def Regional_mask(resolution):

    if resolution == '005':
        xres = 0.005
        yres = 0.005
    elif resolution == '05x0625':
        xres = 0.625
        yres = 0.5
    elif resolution == '1x1':
        xres = 1.0
        yres = 1.0
    else:
        raise ValueError("Resolution not available in Regional_mask")

    file_in='/u/bbyrne1/Rasterized_US_states.nc'
    f=Dataset(file_in,mode='r')
    lat_res = f.variables['lat'+resolution][:]
    lon_res = f.variables['lon'+resolution][:]
    USA_Regions_res = f.variables['mask_regions_'+resolution][:]
    f.close()

    earth_radius = 6371009 # in meters

    lat_dist0 = pi * earth_radius / 180.0
    y = lon_res*0. + yres * lat_dist0
    x= lat_res*0.
    for i in range(np.size(lat_res)):
        x[i]= xres * lat_dist0 * cos(radians(lat_res[i]))
    area_res = np.zeros((np.size(x),np.size(y)))
    for i in range(np.size(y)):
        for j in range(np.size(x)):
            area_res[j,i] = np.abs(x[j]*y[i])

    return lat_res, lon_res, USA_Regions_res, area_res
