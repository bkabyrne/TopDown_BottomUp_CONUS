import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress
import numpy as np

'''
plot_deltaFlux_lateralFlux.py

This program makes a scatter plot of the Posterior-Prior inversion difference 
against the net lateral flux for each region

output:
 - ../Figures/dFlux_lateral.png

'''

Regional_CO2_Budget = pd.read_csv('../Data_processed/Regional_CO2_Budget.csv')

Regional_CO2_Budget['Bottom-up NCE'] = (Regional_CO2_Budget['Biofuel wood'] +
                                        Regional_CO2_Budget['Biofuel ethanol'] +
                                        Regional_CO2_Budget['Biofuel biodiesal'] +
                                        Regional_CO2_Budget['Incineration'] +
                                        Regional_CO2_Budget['FF and IPPU'] +
                                        Regional_CO2_Budget['Crop yield'] +
                                        Regional_CO2_Budget['Livestock Respiration'] +
                                        Regional_CO2_Budget['Human Respiration'] +
                                        Regional_CO2_Budget['Forest Harvest'] +
                                        Regional_CO2_Budget['Forest inventory'] +
                                        Regional_CO2_Budget['grassland stockchange'] +
                                        Regional_CO2_Budget['cropland stockchange'] +
                                        Regional_CO2_Budget['residual wood'] +
                                        Regional_CO2_Budget['PIC and SWDS stockchange'] +
                                        Regional_CO2_Budget['wood trade'] +
                                        Regional_CO2_Budget['residual crop'] +
                                        Regional_CO2_Budget['crop landfill stockchange'] +
                                        Regional_CO2_Budget['crop trade'] +
                                        Regional_CO2_Budget['Lake and River emissions'] +
                                        Regional_CO2_Budget['Lake and River carbon burial'] +
                                        Regional_CO2_Budget['Coastal carbon export'] )

Regional_CO2_Budget['net_lateral_flux'] = ( Regional_CO2_Budget['Biofuel wood'] +
                                            Regional_CO2_Budget['Biofuel ethanol'] +
                                            Regional_CO2_Budget['Biofuel biodiesal'] +
                                            Regional_CO2_Budget['Incineration'] +
                                            Regional_CO2_Budget['Crop yield'] +
                                            Regional_CO2_Budget['Livestock Respiration'] +
                                            Regional_CO2_Budget['Human Respiration'] +
                                            Regional_CO2_Budget['Forest Harvest'] +
                                            Regional_CO2_Budget['residual wood'] +
                                            Regional_CO2_Budget['PIC and SWDS stockchange'] +
                                            Regional_CO2_Budget['wood trade'] +
                                            Regional_CO2_Budget['residual crop'] +
                                            Regional_CO2_Budget['crop landfill stockchange'] +
                                            Regional_CO2_Budget['crop trade'] +
                                            Regional_CO2_Budget['Lake and River emissions'] +
                                            Regional_CO2_Budget['Lake and River carbon burial'] +
                                            Regional_CO2_Budget['Coastal carbon export'] )


change_Flux = Regional_CO2_Budget['LNLGIS median'].values - Regional_CO2_Budget['Prior median'].values

Reg_shortname = ['NW','NGP','MW','SW','SGP','SE','NE']

# Assuming you have your data in Regional_CO2_Budget['net_lateral_flux'].values and change_Flux
x = Regional_CO2_Budget['net_lateral_flux'].values
y = change_Flux

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Generate points for the regression line
x_reg = np.linspace(-150, 150, 100)
y_reg = slope * x_reg + intercept

# Create the plot
fig = plt.figure(10, figsize=(6. * (3.5 / 5.), 5. * (3.5 / 5.)), dpi=300)
ax1 = fig.add_axes([0.17, 0.15, 0.79, 0.8])

# Scatter plot and annotations
for i in range(7):
    plt.text(x[i], y[i], Reg_shortname[i], fontsize=8, va='center', ha='center')

# Regression line
plt.plot(x_reg, y_reg, 'k',linewidth=0.75)
plt.text(-140,140,f'R²={r_value**2:.2f}',ha='left',va='top')

# Other plot elements
plt.xlim([-150, 150])
plt.ylim([-150, 150])
plt.plot([-150, 150], [0, 0], 'k:', linewidth=0.7)
plt.plot([0, 0], [-150, 150], 'k:', linewidth=0.7)
plt.ylabel('Posterior - Prior (TgC year$^{-1}$)')
plt.xlabel('Net lateral flux (TgC year$^{-1}$)')

# Save the figure
plt.savefig('../Figures/dFlux_lateral.png')

