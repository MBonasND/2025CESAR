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
  <img src="https://github.com/user-attachments/assets/afbb332a-1b21-4825-8b46-aedce156fa18" alt="F1-WRFDataPlot" width="600"/>
  <br>
</p>


## Code
Python scripts to train and produce forecasts for the simulated data using both the AE and proposed CESAR models. All necessary packages and functions are loaded within the script and do not need to be loaded separately.

## Workflow
To reproduce results for the simulated 2D Burgers’ equation data from the simulation study, one should first download the two .RData files from the “Data” folder. The next step is to then download the following .R files from the “Code” folder: “data_processing.R”, “deep_functions_physics_BurgerSim.R”, and “forecasting_BurgerSim.R”. A user should save all of the aforementioned file in the same directory, then open the file “forecasting_BurgerSim.R” and run it line by line to reproduce results for each of the methods. This file has lines of code that will load any data or functions from the other downloaded files. 

To reproduce the results for the water field application, one should first request the data as detailed above and preprocess it into a 2D matrix (Time by Space) and subsample the spatial locations as detailed in the manuscript. A user should then download the following two files from the “Code” folder: “deep_functions_physics_WaterApplication.R” and “forecasting_WaterApplication.R”, then open the file “forecasting_WaterApplication.R” and run it line by line to reproduce results for each of the methods. This file has lines of code that will load any functions from the other downloaded files but a user will have to add lines themselves to load their version of the preprocessed application data.

