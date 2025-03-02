import pandas as pd
import numpy as np
from netCDF4 import Dataset
from math import pi, cos, radians
from bisect import bisect_left, bisect_right
import xarray as xr

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
        xres = 0.05
        yres = 0.05
    elif resolution == '05x0625':
        xres = 0.625
        yres = 0.5
    elif resolution == '1x1':
        xres = 1.0
        yres = 1.0
    else:
        raise ValueError("Resolution not available in Regional_mask")

    print()
    file_in='/home/bbyrne/TopDown_BottomUp_CONUS/Data_input/Rasterized_US_states.nc'
    f=Dataset(file_in,mode='r')
    lat_res = f.variables['lat'+resolution][:]
    lon_res = f.variables['lon'+resolution][:]
    if resolution == '005':
        USA_Regions_res = f.variables['mask_regions005'][:]
    else:
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

def define_state_groupings():
    
    Region = {}
    Region['Northwest'] = ['WASHINGTON', 'OREGON', 'IDAHO']
    Region['N Great Plains'] = ['MONTANA', 'WYOMING', 'NORTH DAKOTA', 'SOUTH DAKOTA', 'NEBRASKA']
    Region['Midwest'] = ['MINNESOTA', 'IOWA', 'MISSOURI', 'WISCONSIN', 'ILLINOIS', 'MICHIGAN', 'INDIANA', 'OHIO']
    Region['Southwest'] = ['CALIFORNIA', 'NEVADA', 'UTAH', 'COLORADO', 'NEW MEXICO', 'ARIZONA']
    Region['S Great Plains'] = ['TEXAS', 'OKLAHOMA', 'KANSAS']
    Region['Southeast'] = ['LOUISIANA', 'ARKANSAS', 'MISSISSIPPI', 'ALABAMA', 'GEORGIA', 'FLORIDA', 'SOUTH CAROLINA', 'NORTH CAROLINA', 'VIRGINIA', 'KENTUCKY', 'TENNESSEE']
    Region['Northeast'] = ['WEST VIRGINIA', 'MARYLAND', 'NEW JERSEY', 'DELAWARE', 'NEW YORK', 'CONNECTICUT', 'MASSACHUSETTS', 'VERMONT', 'NEW HAMPSHIRE', 'MAINE', 'RHODE ISLAND', 'DISTRICT OF COLUMBIA','PENNSYLVANIA']

    return Region

def regrid_ARRAY(source, lat_size, lon_size, regrid, lat_regrid_size, lon_regrid_size):
    '''
    Regrids the Flux (Net Ecosystem Exchange) variable from the source grid to a new grid.

    Parameters:
    - source (xarray.Dataset): Input dataset with coordinates 'lat', 'lon' and variables 'Flux' and 'area'.
    - lat_size (float): Original latitude grid size.
    - lon_size (float): Original longitude grid size.
    - regrid (xarray.Dataset): Output dataset with coordinates 'lat', 'lon' and variable 'area'.
    - lat_regrid_size (float): New latitude grid size for the regrid.
    - lon_regrid_size (float): New longitude grid size for the regrid.

    Returns:
    - regrid (xarray.Dataset): Output dataset with regridded 'Flux' added.
    '''

    # Ensure latitude and longitude values are sorted in ascending order
    if np.all(np.diff(source['lat'].values) < 0):
        source = source.sel(lat=source.lat[::-1])
    elif not np.all(np.diff(source['lat'].values) > 0):
        raise ValueError("Source latitudes must be monotonic")

    if np.all(np.diff(source['lon'].values) < 0):
        source = source.sel(lon=source.lon[::-1])
    elif not np.all(np.diff(source['lon'].values) > 0):
        raise ValueError("Source longitudes must be monotonic")

    # Calculate the edges of the latitude and longitude values for the source grid
    lat_values_edges = np.append(source['lat'].values - lat_size / 2., source['lat'].values[-1] + lat_size / 2.)
    lon_values_edges = np.append(source['lon'].values - lon_size / 2., source['lon'].values[-1] + lon_size / 2.)

    # Initialize the output array
    ARRAY_out = np.zeros((np.size(regrid['lat']), np.size(regrid['lon'])))

    # Calculate the midpoints and edges for the regrid
    regrid_lat_midpoints = regrid['lat'].values
    regrid_lon_midpoints = regrid['lon'].values
    out_top_grid = regrid_lat_midpoints + lat_regrid_size / 2.
    out_bottom_grid = regrid_lat_midpoints - lat_regrid_size / 2.
    out_left_grid = regrid_lon_midpoints - lon_regrid_size / 2.
    out_right_grid = regrid_lon_midpoints + lon_regrid_size / 2.

    # Check if regrid midpoints are within the valid range of the source grid
    if (np.any(out_bottom_grid < lat_values_edges[0]) or np.any(out_top_grid > lat_values_edges[-1]) or
        np.any(out_left_grid < lon_values_edges[0]) or np.any(out_right_grid > lon_values_edges[-1])):
        print("Warning: Some regrid midpoints are out of the valid range of source latitudes and longitudes")
        # Adjust out-of-range midpoints to be within the valid range
        out_bottom_grid = np.clip(out_bottom_grid, lat_values_edges[0], lat_values_edges[-1])
        out_top_grid = np.clip(out_top_grid, lat_values_edges[0], lat_values_edges[-1])
        out_left_grid = np.clip(out_left_grid, lon_values_edges[0], lon_values_edges[-1])
        out_right_grid = np.clip(out_right_grid, lon_values_edges[0], lon_values_edges[-1])

    # Loop through each regrid cell to calculate the regridded Flux
    for ii in range(len(regrid_lat_midpoints)):
        for jj in range(len(regrid_lon_midpoints)):
            lat_start = bisect_left(lat_values_edges, out_bottom_grid[ii])
            lat_end = bisect_right(lat_values_edges, out_top_grid[ii])
            lon_start = bisect_left(lon_values_edges, out_left_grid[jj])
            lon_end = bisect_right(lon_values_edges, out_right_grid[jj])

            # Adjust indices if they are out of bounds
            if lat_start == lat_end:
                lat_start = max(0, lat_start - 1)
                lat_end = min(len(lat_values_edges), lat_end + 1)
            if lon_start == lon_end:
                lon_start = max(0, lon_start - 1)
                lon_end = min(len(lon_values_edges), lon_end + 1)

            Temp_total_flux = 0.
            Temp_total_area = 0.
            for iit in range(lat_start - 1, lat_end):
                for jjt in range(lon_start - 1, lon_end):
                    if 0 <= iit < np.size(source['lat'].values) and 0 <= jjt < np.size(source['lon'].values):
                        in_lat_grid_center = source['lat'].values[iit]
                        in_lon_grid_center = source['lon'].values[jjt]
                        in_top_grid = in_lat_grid_center + lat_size / 2.
                        in_bottom_grid = in_lat_grid_center - lat_size / 2.
                        in_left_grid = in_lon_grid_center - lon_size / 2.
                        in_right_grid = in_lon_grid_center + lon_size / 2.

                        # Calculate the overlapping box area between source and regrid cells
                        bottom_box = np.max([in_bottom_grid, out_bottom_grid[ii]])
                        top_box = np.min([in_top_grid, out_top_grid[ii]])
                        left_box = np.max([in_left_grid, out_left_grid[jj]])
                        right_box = np.min([in_right_grid, out_right_grid[jj]])

                        rough_area_small_box = (top_box - bottom_box) * (right_box - left_box)
                        rough_source_area = (in_top_grid - in_bottom_grid) * (in_right_grid - in_left_grid)

                        if rough_source_area == 0:
                            print(f"Warning: rough_source_area is zero at source indices ({iit}, {jjt})")

                        # Accumulate the total flux and area for the regrid cell
                        Temp_total_flux += source['Flux'].values[iit, jjt] * source['area'].values[iit, jjt] * (rough_area_small_box / rough_source_area)
                        Temp_total_area += source['area'].values[iit, jjt] * (rough_area_small_box / rough_source_area)

            # Calculate the regridded Flux for the current regrid cell
            if Temp_total_area > 0:
                ARRAY_out[ii, jj] = Temp_total_flux / Temp_total_area
            else:
                print(f"Warning: Temp_total_area is zero for regrid cell ({ii}, {jj})")

    # Print total Flux values for source and regridded data for verification
    print('=== Check totals ===')
    total_source_Flux = np.sum(source['Flux'].values * source['area'].values) * 1e-12
    total_regrid_Flux = np.sum(ARRAY_out * regrid['area'].values) * 1e-12
    ratio = total_source_Flux / total_regrid_Flux

    print(f"Total source Flux: {total_source_Flux} TgC")
    print(f"Total regridded ARRAY_out * regrid['area']: {total_regrid_Flux} TgC")
    print(f"Ratio: {ratio}")

    # Add the regridded Flux to the regrid dataset
    regrid['Flux'] = (('lat', 'lon'), ARRAY_out)

    return regrid
