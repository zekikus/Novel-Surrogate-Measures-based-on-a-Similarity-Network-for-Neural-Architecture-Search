import numpy as np

CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

def path_distance(cell_1, cell_2):
    """ 
    compute the distance between two architectures
    by comparing their path encodings
    """
    return np.sum(np.array(cell_1.encode('path') != np.array(cell_2.encode('path'))))

def caz_encoding_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their in-out edges and path encodings
    """
    return np.sum(cell_1.encode('caz') != cell_2.encode('caz')) + path_distance(cell_1, cell_2)

def jackard_distance_caz(cell_1, cell_2):
    """
    compute the jackard distance between two architectures
    by comparing their caz encodings (in-out edges + path encoding - Tanimoto Index)
    """

    # Cell 1 - Path encoding, Tanimoto Index (Vector with 364 elements)
    cell1_path_vct = np.array(cell_1.encode('path'))
    # Cell 2 - Path encoding
    cell2_path_vct = np.array(cell_2.encode('path'))
    # Cell 1 - In-out edges encoding - Can Hoca'nın Path Encoding üzerine önerdiği encoding
    cell1_caz_vct = np.array(cell_1.encode('caz'))
    # Cell 2 - In-out edges encoding
    cell2_caz_vct = np.array(cell_2.encode('caz'))

    # Compute the jackard distance
    jk_dist = np.sum(cell1_path_vct * cell2_path_vct) + np.sum(cell1_caz_vct * cell2_caz_vct)
    total_hamming_dist = np.sum(cell1_caz_vct != cell2_caz_vct) + np.sum(cell1_path_vct != cell2_path_vct)
    return total_hamming_dist / (total_hamming_dist + jk_dist)