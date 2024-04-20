import utils
import pandas as pd
import matplotlib.pyplot as plt

'''
plot_flux_vs_population.py

This program calculates regional population totals and then plots regional fossil fuel 
emissions against population and regional biofuel emissions against population

output:
 - ../Figures/FF_vs_Biofuels_vs_pop.png
'''

# Short name for regions
Reg_shortname = ['NW','NGP','MW','SW','SGP','SE','NE']

# Read state & regional info                                                                                                                                                
states = utils.create_state_dataframe()
regions = utils.define_state_groupings()

# Read State Population data
file_path = '/u/bbyrne1/python_codes/TopDown_BottomUp_CONUS/Data_input/NST-EST2023-POP.csv'
df = pd.read_csv(file_path)
df['Region'] = df['Region'].str.replace('.', '').str.upper()
df['2020'] = df['2020'].str.replace(',', '').fillna(0).astype(int)

# Calculate Regional population
Regional_pop_dict= {}
for region in regions:
    Regional_pop_dict[region] = df[df['Region'].isin(regions[region])].iloc[:,2].sum(axis=0)
Regional_pop = pd.DataFrame.from_dict(Regional_pop_dict, orient='index')

# Read Regional CO2 Budget 
Regional_CO2_Budget = pd.read_csv('../Data_processed/Regional_CO2_Budget.csv')
Biofuels = Regional_CO2_Budget['Biofuel wood'].values + Regional_CO2_Budget['Biofuel ethanol'].values + Regional_CO2_Budget['Biofuel biodiesal'].values

# --- Create figure ---
fig = plt.figure(1,figsize=(12.*(3.5/5.),5.*(3.5/5.)),dpi=300)
# Plot regional Fossil Fuel vs population
ax1 = fig.add_axes([0.15/2.,0.15,0.8/2.,0.8])
for i in range(7):
    plt.text( Regional_pop.values[i] * 1e-6 , Regional_CO2_Budget['FF and IPPU'].values[i] ,Reg_shortname[i])
plt.xlim([0,100])
plt.ylim([0,400])
plt.text(100.*0.01,400.*0.99,'(a)',ha='left',va='top')
plt.ylabel('FF and IPPU (TgC year$^{-1}$)')
plt.xlabel('Population')
# Plot regional Biofuel vs population
ax1 = fig.add_axes([1.15/2.,0.15,0.8/2.,0.8])
for i in range(7):
    plt.text( Regional_pop.values[i] * 1e-6 , Biofuels[i] ,Reg_shortname[i])
plt.xlim([0,100])
plt.ylim([0,40])
plt.text(100.*0.01,40.*0.99,'(b)',ha='left',va='top')
plt.ylabel('Biofuels (TgC year$^{-1}$)')
plt.xlabel('Population')
plt.savefig('../Figures/FF_vs_Biofuels_vs_pop.png')
