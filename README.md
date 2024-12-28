# Machine Learning Approaches for Developing Potential Surfaces: Applications to OH-(H2O)n (n=1-3) Complexes

## Overview
This repository includes the codes and results for the manuscript:
***Machine Learning Approaches for Developing Potential Surfaces: Applications to OH-(H2O)n (n=1-3) Complexes*** [link](xxxxx)


Due to the limited space on GitHub, our [Zenodo repository](https://zenodo.org/records/14563580) contains all the training and test data (structures and labels)

## Content list
 
* [mobml_reference_data](mobml_reference_data): Please see the Zenodo repository. The reference structures for each system is zipped under each zip file. The predicted energies for the test systems are also within the zip files. The reference electronic structure results are under [csvs](reference_data/csvs) folder and labeled with the corresponding systems. 

* [dmc_data](dmc_data): Please see the Zenodo repository. TODO: Greta please put your DMC data under this folder and write a description. I leave the files as what it is on google drive and not sure if they are useful. if not, please delete them.

* [dmc_model](dmc_model): The final versions of the neural network models used to obtain the DMC results for the four systems included in the manuscript. For each system, there is a .pth file containing the neural network model itself, plus a .py script containing the code for the system's associated molecular descriptor, which calls the NN model in the context of a DMC simulation.

* [plot.ipynb](plot.ipynb): Plot the results and make the figures in the manuscript. 

* [lc_data.h5](lc_data.h5): Save the learning curve accuracy that could be retrived by the [plot.ipynb](plot.ipynb).


## Please cite us as

```
@article{xxx,
  title={xxxx},
  author={xxxx},
  doi = {xxx},
  journal = {xxxx},
}
```
