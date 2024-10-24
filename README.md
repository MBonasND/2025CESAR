# 2025CESAR
Supplemental codes for "CESAR: A Convolutional Echo State AutoencodeR for High-Resolution Wind Power Forecasting" by Matthew Bonas, Paolo Giani, Paola Crippa, and Stefano Castruccio

## Data
Folder containing a .pkl file called `burger2d.pkl` with a single simulation of the 2D Burgers' equation data used throughout the associated manuscript. 

<p align = "center">
  <img src="https://github.com/user-attachments/assets/f60151c9-db63-4066-aa45-aeec7d4c637d" alt="F2-Burger2DSim0" width="600"/>
  <br>
</p>

The data used for the application are generated using the WRF model and its descriptions can be found in [Giani et. al, 2022](https://journals.ametsoc.org/view/journals/mwre/150/5/MWR-D-21-0216.1.xml).

<p align = "center">
  <img src="https://github.com/user-attachments/assets/d959a5bb-67c1-48fe-8cf2-2b6cc811ae16" alt="F1-WRFDataPlot" width="600"/>
  <br>
</p>

## Code
Python scripts to train and produce forecasts for the simulated data using both the AE and proposed AE-ESN models. All necessary packages and functions are loaded within the script and do not need to be loaded separately.
