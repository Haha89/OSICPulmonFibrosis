## OSIC Pulmonary Fibrosis Progression

Kaggle : https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

The aim of this competition is to predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (FVC), i.e. the volume of air exhaled.

In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time Week = 0 and has numerous follow up visits over the course of approximately 1-2 years, at which time their FVC is measured.

In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. You are asked to predict the final three FVC measurements for each patient, as well as a confidence value in your prediction.
Since this is real medical data, you will notice the relative timing of FVC measurements varies widely. The timing of the initial measurement relative to the CT scan and the duration to the forecasted time points may be different for each patient. This is considered part of the challenge of the competition. To avoid potential leakage in the timing of follow up visits, you are asked to predict every patient's FVC measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

## Projet's architecture
You will need the dataset located : https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data

osic-pulmonary-fibrosis-progression/
- data/

	- models/: contains the trained network
	- OSIC-EDA.html: file used for exploratory data analysis

		
- lib/
	- data_analysis.py: script to generate charts and EDA
	- utils.py: bunch of useful functions used throughout the scripts
	- ODE_network.py: contains the definition of our neural network
	- dataset.py: contains the dataset definition
	- training_ode and full_trainging: training scripts (with and without test/val)
	- script_submission : script to create the submission.csv file
	- batch_3d_scan: script to preprocess the 3d-scans from training dataset
	- scan_preprocessing.py: contains functions to preprocess the 3d-scans
	- examples-001 to 005.py: some test scripts
