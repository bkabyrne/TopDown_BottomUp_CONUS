import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
from pylab import *
import numpy as np
import utils

'''

plot_total_bottom_up_stockchange.py

This program makes a barplot of the the bottom-up stockchange estimates

output:
 - ../Figures/CONUS_bottom_up_stockchange.png

'''

Regional_CO2_Budget = pd.read_csv('../Data_processed/Regional_CO2_Budget.csv')

CONUS_CO2_Budget = Regional_CO2_Budget.sum(axis=0)

CONUS_CO2_Budget_total = ( CONUS_CO2_Budget['Forest inventory'] + CONUS_CO2_Budget['grassland stockchange'] +
                           CONUS_CO2_Budget['cropland stockchange'] + CONUS_CO2_Budget['PIC and SWDS stockchange'] +
                           CONUS_CO2_Budget['crop landfill stockchange'] + CONUS_CO2_Budget['Lake and River carbon burial'])



Inv_cmap_test1 = cm.get_cmap('viridis')
Inv_colorst = Inv_cmap_test1(np.linspace(0, 1.0, 7))


fig = plt.figure(2, figsize=(6.75*0.95,3.5*0.95), dpi=300)
ax1 = fig.add_axes([0.15,0.21,0.825,0.77])
plt.fill_between([0,1],[0,0],[CONUS_CO2_Budget_total,CONUS_CO2_Budget_total],color='green', edgecolor='black',linewidth=0.5)
plt.fill_between([1,2],[0,0],[CONUS_CO2_Budget['Forest inventory'],CONUS_CO2_Budget['Forest inventory']],color='green', edgecolor='black',linewidth=0.5,alpha=0.5)
plt.fill_between([2,3],[0,0],[CONUS_CO2_Budget['grassland stockchange'],CONUS_CO2_Budget['grassland stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.5)
plt.fill_between([3,4],[0,0],[CONUS_CO2_Budget['cropland stockchange'],CONUS_CO2_Budget['cropland stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.5)
plt.fill_between([4,5],[0,0],[CONUS_CO2_Budget['PIC and SWDS stockchange'],CONUS_CO2_Budget['PIC and SWDS stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.5)
plt.fill_between([5,6],[0,0],[CONUS_CO2_Budget['crop landfill stockchange'],CONUS_CO2_Budget['crop landfill stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.5)
plt.fill_between([6,7],[0,0],[CONUS_CO2_Budget['Lake and River carbon burial'],CONUS_CO2_Budget['Lake and River carbon burial']],color='green', edgecolor='black',linewidth=0.5,alpha=0.5)
plt.xlim([-0.7,7.25])
plt.xticks(np.arange(7)+0.5)
ax1.set_xticklabels(['$\mathrm{ \Delta C_{Total} }$',
                     '$\mathrm{ \Delta C_{Forest} }$',
                     '$\mathrm{ \Delta C_{Grassland} }$',
                     '$\mathrm{ \Delta C_{Cropland} }$',
                     'PIC and SWDS',
                     'landfill crop',
                     'aquatic burial'],rotation=20)
plt.ylim([-350,25])
plt.arrow(-0.55,-25,0,-300,width=0.0015,head_width=0.2,head_length=6.5,fc='k')
plt.text(-0.45,(-25.-325.)/2.,'increasing stocks',rotation=270,ha='left',va='center')
plt.ylabel('CONUS carbon stockchange\n(TgC year$^{-1}$)')
plt.plot([-0.7,7.25],[0,0],'k',linewidth=0.3)
# Save figure
plt.savefig('../Figures/CONUS_bottom_up_stockchange.png', dpi=300)

