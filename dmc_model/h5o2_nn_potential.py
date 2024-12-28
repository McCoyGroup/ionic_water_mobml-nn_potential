import torch
import torch.nn as nn
import numpy as np

input_size = 21
hidden_size = 150
output_size = 1

model = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.BatchNorm1d(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(),
                    nn.Linear(hidden_size, output_size,bias=True),
                    nn.ReLU()

)

# Load the model's state dictionary from the saved file
model.load_state_dict(torch.load('h5o2_nn_model_molec_sorted_1.5mill_0_reg_150hidden_bn.pth',map_location=torch.device('cpu')))

# Put the model in evaluation mode
model.eval()

def calculate_feature_vectors(molecule_coords, atomic_numbers):
    """
    Calculate the feature vectors using the lower triangle of the molecule sorted Coulomb matrix for H3O+(H2O) (aka h5o2).
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, 7, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (7,) containing the atomic numbers.

    Returns:
    np.ndarray: An array of input features for the nn model of shape (n_molecules, 21).
    """
    
    n_molecules, n_atoms, _ = molecule_coords.shape
    
    # Initialize the Coulomb matrices
    coulomb_matrices = np.zeros((n_molecules, n_atoms, n_atoms))
    
    # Compute pairwise distance matrices
    distance_matrices = np.linalg.norm(molecule_coords[:, :, np.newaxis] - molecule_coords[:, np.newaxis, :], axis=-1)
    
    # Calculate off-diagonal elements
    Z_product = atomic_numbers[:, np.newaxis] * atomic_numbers[np.newaxis, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        coulomb_matrices = Z_product / distance_matrices
        coulomb_matrices[distance_matrices == 0] = 0  # Handle division by zero
    
    # Calculate diagonal elements
    diagonal_elements = 0.5 * atomic_numbers**2.4
    np.einsum('ijj->ij', coulomb_matrices)[:] = diagonal_elements
    
    groups = [[0],[1,2,3],[4,5,6]] #assuming the input geometries list the shared proton first
    
    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    
    for a in range(n_molecules):
        
        norms = []
        for g in groups:
            norms.append(np.sum([np.linalg.norm(coulomb_matrices[a][i]) for i in g]))
            
        sorted_indices = np.argsort(norms)[::-1]

        new_index = []
        for i in sorted_indices:
            for j in groups[i]:
                new_index.append(j)

        sorted_CM = coulomb_matrices[a][:, new_index][new_index, :]
        
        lower_tri = np.tril_indices(n_atoms, -1)
        
        lower_triangles[a] = sorted_CM[lower_tri]
    
    return lower_triangles

def cart_to_pot(cds):
    """
    Calculates the potential energies of input geometries for H3O+(H2O) (aka h5o2) in a.u.
    for use in DMC simulations

    Parameters:
    cds (np.ndarray): An array of shape (n_molecules, 7, 3) containing the Cartesian coordinates in Bohr.

    Returns:
    np.ndarray: An array shape (n_molecules, ) of potenial energies in a.u..
    """
    atoms = np.array([1,1,1,8,1,1,8])
    
    features = calculate_feature_vectors(cds,atoms)
      
    energy = model(torch.tensor(np.array(features),dtype=torch.float32))
    #convert output back to unshifted energy in a.u.
    energy_unshifted = torch.tensor([(10**(i)-100)/219474.63136320 for i in energy])   
    return energy_unshifted.detach().numpy().reshape(len(cds))

