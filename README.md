# cmepda_project
The aim of this project is to implement a machine learning algorithm to improve the Missing Energy (MET) resolution in LHC experiments using CERN Open Data.

## Requirements
To run this code you may need to install the following packages.

```bash
pip install xgboost
pip install uproot
```

Alternatively, since we are not using an NVIDIA GPU you can install the lighter xgboost package. This is not recommended since some functionalities may not work.

```bash
pip install xgboost-cpu
```

## Exploring the dataset
In the data_retrieving.C file we read the Open Data .root file, particularly exploring the branches inside the 'Event' TTree. This is useful to decide the entries to keep in order to pass them as parameters of the ML algorithm.
On this note data_skimming.C only keeps MET and Jet entries, excluding trigger and flag bits.

## The Machine Learning algorithm
The MET resolution is improved through a ML regression algorithm, using XGBoost packages.
