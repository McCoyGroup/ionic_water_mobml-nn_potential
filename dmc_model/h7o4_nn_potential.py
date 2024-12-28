import torch
import torch.nn as nn
import numpy as np

input_size = 55
hidden_size = 540
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

#load the model's state dictionary from the saved file
model.load_state_dict(torch.load('h7o4_nn_model_molec_atom_sorted_2mill_1e-6_reg_540hidden_bn.pth',map_location=torch.device('cpu')))

# Put the model in evaluation mode
model.eval()

def calculate_feature_vectors(molecule_coords, atomic_numbers):
    """
    Calculate the feature vectors using the lower triangle of the molecule + atom sorted Coulomb matrix for OH-(H2O)3 (aka h7o4).
    
    Parameters:
    molecule_coords (np.ndarray): An array of shape (n_molecules, 11, 3) containing the Cartesian coordinates.
    atomic_numbers (np.ndarray): An array of shape (11,) containing the atomic numbers.

    Returns:
    np.ndarray: An array of input features for the nn model of shape (n_molecules, 55).
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
    
    groups = [[0,1],[2,3,4],[5,6,7],[8,9,10]] #assuming the hydroxide ion is listed first in the coordinates
    group_sizes = [len(i) for i in groups]
    group_starts = [i[0] for i in groups]
    
    lower_triangles = np.zeros((n_molecules, n_atoms * (n_atoms - 1) // 2))
    
    for a in range(n_molecules):
        reorder = []
        group_norms = []
        for i in range(len(group_sizes)):
            start = group_starts[i]
            end = start+group_sizes[i]
            reorder.append(np.argsort([np.linalg.norm(coulomb_matrices[a][j]) for j in range(start,end)])+start)
            
            group_norm = np.sum([np.linalg.norm(coulomb_matrices[a][j]) for j in range(start,end)])
            group_norms.append(group_norm)

        group_order = np.argsort(group_norms)

        sorted_indices = []
        for i in range(len(group_sizes)):
            sorted_indices.append(reorder[group_order[i]])

        sorted_CM = coulomb_matrices[a][:, np.concatenate(sorted_indices)][np.concatenate(sorted_indices), :]

        lower_tri = np.tril_indices(n_atoms, -1)
        
        lower_triangles[a] = sorted_CM[lower_tri]
    
    return lower_triangles

def cart_to_pot(cds):
    """
    Calculates the potential energies of input geometries for OH-(H2O)3 (aka h7o4) in a.u.
    for use in DMC simulations

    Parameters:
    cds (np.ndarray): An array of shape (n_molecules, 11, 3) containing the Cartesian coordinates in Bohr.

    Returns:
    np.ndarray: An array shape (n_molecules, ) of potenial energies in a.u..
    """
    atoms = np.array([8,1,8,1,1,8,1,1,8,1,1])
        
    features = calculate_feature_vectors(cds,atoms)

    energy = model(torch.tensor(np.array(features),dtype=torch.float32))
    #convert output back to unshifted energy in a.u.
    energy_unshifted = torch.tensor([(10**(i)-100)/219474.63136320 for i in energy])   
    return energy_unshifted.detach().numpy().reshape(len(cds))


