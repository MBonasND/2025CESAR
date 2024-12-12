# 2025CESAR
Supplemental codes for "CESAR: A **C**onvolutional **E**cho **S**tate **A**utoencode**R** for High-Resolution Wind Power Forecasting" by Matthew Bonas, Paolo Giani, Paola Crippa, and Stefano Castruccio

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
To reproduce results for the simulated 2D Burgers’ equation data from the simulation study, one should first download the file `burger2d.pkl` file from the “Data” folder. The next step is to then download the following three files from the “Code” folder: `Burger2D_CAE.py`, `Burger2D_CESAR.py` and `functions.py`. A user should save all of the aforementioned file in the same directory. To reproduce the results from the manuscript, a user should first run `Burger2D_CAE.py` to train the CAE portion of the method and then run `Burger2D_CESAR.py` to produce train and forecasts for the proposed CESAR approach. 

To reproduce the results for the WRF wind speed application, one should first generate the data using the descriptions from [Giani et. al, 2022](https://journals.ametsoc.org/view/journals/mwre/150/5/MWR-D-21-0216.1.xml). A user should then download the same files used for the simulated 2D Burgers' equation data. It would then be required to adjust the scripts to now load this newly generated data and to modify the hyper-parameters from the CAE and CESAR methods to be those used in the manuscript. 

