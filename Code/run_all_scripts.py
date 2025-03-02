import subprocess

print('--- Aggregate Data ---')
subprocess.run(['python', './aggregate_tabular_carbon_datasets.py'])
subprocess.run(['python', './aggregate_agricultural_inventory.py'])
subprocess.run(['python', './aggregate_TopDown_regional.py'])

print('--- Create CO2 Budget ---')
subprocess.run(['python', './combined_topdown_and_bottomup.py'])

print('--- Create Plot ---')
subprocess.run(['python', './plot_regional_budget_on_map.py'])

