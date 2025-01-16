# Machine Learning Approaches for Developing Potential Surfaces: Applications to OH-(H2O)n (n=1-3) Complexes

## Overview
This repository includes the codes and results for the manuscript:
***Machine Learning Approaches for Developing Potential Surfaces: Applications to OH-(H2O)n (n=1-3) Complexes*** [link](xxxxx)


Due to the limited space on GitHub, our [Zenodo repository](https://zenodo.org/records/14563580) contains all the training and test data (structures and labels)

## Content list
 
* [mobml_reference_data](mobml_reference_data): Please see the Zenodo repository. The reference structures for each system is zipped under each zip file. The predicted energies for the test systems are also within the zip files. The reference electronic structure results are under [csvs](reference_data/csvs) folder and labeled with the corresponding systems. 

* [dmc_data](dmc_data): Please see the Zenodo repository. The full training and test sets of structures and energies used for fitting the NN+(MOB-ML) model for each system is zipped under the zip file. Each file is a .npz file where the dictionary key for the structures (in Bohr) is 'cds', and the key for the corresponding energies with respect to the minimum energy structure of the MOB-ML model (in cm-1) is 'energies'.

* [dmc_model](dmc_model): The final versions of the neural network models used to obtain the DMC results for the four systems included in the manuscript. For each system, there is a .pth file containing the neural network model itself, plus a .py script containing the code for the system's associated molecular descriptor, which calls the NN model in the context of a DMC simulation.

* [plot.ipynb](plot.ipynb): Plot the results and make the figures in the manuscript. 

* [lc_data.h5](lc_data.h5): Save the learning curve accuracy that could be retrived by the [plot.ipynb](plot.ipynb).


## Please cite us as

```
@article{jacobsone2024,
  title={Machine Learning Approaches for Developing Potential Surfaces: Applications to OH-(H2O)n (n=1-3) Complexes},
  author={Jacobson, Greta and Cheng, Lixue and Bhethanabotla, Vignesh and Sun, Jiace and McCoy, Anne},
  doi = {10.26434/chemrxiv-2024-5c808-v2},
  journal = {ChemRxiv},
}
```
