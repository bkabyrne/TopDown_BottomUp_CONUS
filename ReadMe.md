# Read me for CONUS study

This repository contains python codes used in Byrne et al. which investigates the regional carbon budget within CONUS using both top-down and bottom-up methods.


Reference:
Byrne et al. in prep

# Code:

run_all_scripts.py
  - Runs all scripts to aggregate data, create dataset and plot

## Aggregateion scripts
  - aggregate_tabular_carbon_datasets.py
     - This combines all of the tabular data into state totals
  - aggregate_agricultural_inventory.py
     - Calculates regional totals in cropland and grassland carbon stock changes
  - aggregate_TopDown_regional.py
     - Calculates regional NCE totals from the v10 OCO-2 MIP

## Dataset creation
  - combined_topdown_and_bottomup.py
     - This combines the aggregated datasets into a single table of the CO2 budget for regions across CONUS

## Plotting scripts
  - plot_regional_budget_on_map.py
     - Generates a plot of the regional carbon budgets embedded on a map of CONUS
  - plot_OCO2MIP_experiments_regional_budget_on_map.py
     - Makes regional NCE plots for all inversion experiments on bottom-up estimate
  - plot_CONUS_budget.py
     - Creates a bar plot of CONUS totals for LNLGIS and all bottom-up fluxes
  - plot_regional_climate.py
     - Temperature & precipitation anomaly plots
  - calc_topdown_regional_covariances.py
     - Inversion regional correlation matrix calculation and plots
  - plot_total_bottom_up_stockchange.py
     - This program makes a barplot of the the bottom-up stockchange estimates
  - plot_deltaFlux_lateralFlux.py
     - Scatter plot of the Posterior-Prior difference vs net lateral flux
  - plot_flux_vs_population.py
     - regional fossil fuel/biofuel emissions against population 
