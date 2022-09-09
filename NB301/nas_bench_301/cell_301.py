import numpy as np
import random
from collections import namedtuple
from sklearn.linear_model import LinearRegression

import nasbench301 as nb

OPS = ['max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
       ]
NUM_VERTICES = 4
NUM_OPS = 7
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'
OUTPUT = 'c_k'


class Cell301:

    def __init__(self, arch, arch_cont):
        self.arch = arch
        self.arch_cont = arch_cont
        self.solNo = None
        self.fitness = 0
        self.trueFitness = None # Korelasyon hesabı için
        self.fitnessType = None
        self.reliability = None
        self.upperLimit = 1
        self.reliabilityHistory = []
        self.fitnessHistory = []
        self.regression_model = None
        self.regression_sample_size = 0
        self.estimation_type = None

    def serialize(self):
        return tuple([tuple([tuple(pair) for pair in cell]) for cell in self.arch])

    def get_hash(self):
        return self.serialize()

    def fit_regression_model(self, X, y):
        self.regression_sample_size = len(X)
        self.regression_model = LinearRegression().fit(X, y)  
        
    def get_val_loss(self, nasbench, deterministic=True, patience=50, epochs=None, dataset=None):

        genotype = self.convert_to_genotype(self.arch)
        acc = nasbench[0].predict(config=genotype, representation="genotype")
        return acc / 100
    
    def get_tr_time(self, nasbench, deterministic=True, patience=50, epochs=None, dataset=None):

        genotype = self.convert_to_genotype(self.arch)
        time = nasbench[1].predict(config=genotype, representation="genotype")
        return time
    
    def get_test_loss(self, nasbench, patience=50, epochs=None, dataset=None):
        # currently only val_loss is supported. Just return the val loss here
        return self.get_val_loss(nasbench)
        
    def convert_to_genotype(self, arch):

        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        op_dict = {
        0: 'max_pool_3x3',
        1: 'avg_pool_3x3',
        2: 'skip_connect',
        3: 'sep_conv_3x3',
        4: 'sep_conv_5x5',
        5: 'dil_conv_3x3',
        6: 'dil_conv_5x5'
        }

        darts_arch = [[], []]
        i=0
        for cell in arch:
            for n in cell:
                darts_arch[i].append((op_dict[n[1]], n[0]))
            i += 1
        genotype = Genotype(normal=darts_arch[0], normal_concat=[2,3,4,5], reduce=darts_arch[1], reduce_concat=[2,3,4,5])
        return genotype

    def make_mutable(self):
        # convert tuple to list so that it is mutable
        arch_list = []
        for cell in self.arch:
            arch_list.append([])
            for pair in cell:
                arch_list[-1].append([])
                for num in pair:
                    arch_list[-1][-1].append(num)
        return arch_list
    
    def encode(self, predictor_encoding, type=None, nasbench=None, deterministic=True, cutoff=None):

        if predictor_encoding == 'path':
            return self.encode_paths()
        elif predictor_encoding == 'caz': # Ayla Hoca'nın önerdiği yapı
            return self.caz_encoding(type=type)
        elif predictor_encoding == 'trunc_path':
            if not cutoff:
                cutoff = 100
            return self.encode_paths(cutoff=cutoff)
        elif predictor_encoding == 'adj':
            return self.encode_adj()

        else:
            print('{} is not yet implemented as a predictor encoding \
             for nasbench301'.format(predictor_encoding))
            raise NotImplementedError()

    def distance(self, other, dist_type, cutoff=None):

        if dist_type == 'path':
            return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))
        elif dist_type == 'adj':
            return np.sum(np.array(self.encode_adj() != np.array(other.encode_adj())))
        elif dist_type == 'caz':
            return self.caz_distance(other)
        else:
            print('{} is not yet implemented as a distance for nasbench301'.format(dist_type))
            raise NotImplementedError()

    @classmethod
    def random_cell(cls, 
                    nasbench, 
                    random_encoding, 
                    cutoff=None,
                    max_edges=10, 
                    max_nodes=8,
                    index_hash=None):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts

        if random_encoding != 'adj':
            print('{} is not yet implemented as a mutation \
                encoding for nasbench301'.format(random_encoding))
            raise NotImplementedError()

        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(len(OPS)), NUM_VERTICES)

            #input nodes for conv
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            #input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

        return {'arch': (normal, reduction)}

    def perturb(self, nasbench, edits=1):
        return self.mutate()

    def get_paths(self):
        """ return all paths from input to output """

        path_builder = [[[], [], [], []], [[], [], [], []]]
        paths = [[], []]

        for i, cell in enumerate(self.arch):
            for j in range(len(OPS) + 1):
                if cell[j][0] == 0:
                    path = [INPUT_1, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)
                elif cell[j][0] == 1:
                    path = [INPUT_2, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)
                else:
                    for path in path_builder[i][cell[j][0] - 2]:
                        path = [*path, OPS[cell[j][1]]]
                        path_builder[i][j//2].append(path)
                        paths[i].append(path)

        return paths

    def get_path_indices(self, long_paths=True):
        """
        compute the index of each path
        There are 4 * (8^0 + ... + 8^4) paths total
        If long_paths = False, we give a single boolean to all paths of
        size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
        """
        paths = self.get_paths()
        normal_paths, reduce_paths = paths
        num_ops = len(OPS)
        """
        Compute the max number of paths per input per cell.
        Since there are two cells and two inputs per cell, 
        total paths = 4 * max_paths
        """

        max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    
        path_indices = []

        # set the base index based on the cell and the input
        for i, paths in enumerate((normal_paths, reduce_paths)):
            for path in paths:
                index = i * 2 * max_paths
                if path[0] == INPUT_2:
                    index += max_paths

                # recursively compute the index of the path
                for j in range(NUM_VERTICES + 1):
                    if j == len(path) - 1:
                        path_indices.append(index)
                        break
                    elif j == (NUM_VERTICES - 1) and not long_paths:
                        path_indices.append(2 * (i + 1) * max_paths - 1)
                        break
                    else:
                        index += num_ops ** j * (OPS.index(path[j + 1]) + 1)

        return tuple(path_indices)

    def encode_paths(self, cutoff=None):
        # output one-hot encoding of paths
        path_indices = self.get_path_indices() # Normal Cell + Reduce Cell indices
        num_ops = len(OPS)

        max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    

        path_encoding = np.zeros(4 * max_paths)
        for index in path_indices:
            path_encoding[index] = 1
        if cutoff:
            path_encoding = path_encoding[:cutoff]
        return path_encoding
    
    # Ayla Hoca'nın önerdiği encoding
    def caz_encoding(self, type):
        id = 0 if type == 'normal' else 1
        arch = self.arch[id]
        encoding = np.zeros(56)
        extends = np.zeros(98)
        start_idx = {0: 0, 1: 28, 2: 56, 3: 77, 4: 91}
        first_node = {0: 2, 1: 2, 2: 3, 3: 4, 4: 5}

        # Iterate Node 2, 3, 4, 5
        for idx in range(NUM_VERTICES):
            encoding[(idx * NUM_OPS) + arch[idx * 2][1]] = 1
            encoding[((idx + 1) * NUM_OPS) + arch[idx * 2 + 1][1]] = 1

            in_edge = idx + 2
            out_edge_1 = arch[idx * 2][1]
            out_edge_2 = arch[idx * 2 + 1][1]
            out_node_1 = arch[idx * 2][0]
            out_node_2 = arch[idx * 2 + 1][0]
            
            extends[start_idx[out_node_1] + ((in_edge - (first_node[out_node_1])) * NUM_OPS) + out_edge_1] = 1
            extends[start_idx[out_node_2] + ((in_edge - (first_node[out_node_2])) * NUM_OPS) + out_edge_2] = 1

        return np.concatenate((encoding, extends))
    
    def caz_distance(self, other):
        # Cell 1 - Path encoding, Tanimoto Index
        cell1_path_vct = np.array(self.encode('path'))
        # Cell 2 - Path encoding
        cell2_path_vct = np.array(other.encode('path'))
        # Cell 1 - Path encoding expansion
        cell1_caz_normal_vct = np.array(self.encode('caz', 'normal'))
        cell1_caz_reduce_vct = np.array(self.encode('caz', 'reduce'))
        cell1_caz_vct = np.concatenate((cell1_caz_normal_vct, cell1_caz_reduce_vct))
        # Cell 2 - Path encoding expansion
        cell2_caz_normal_vct = np.array(other.encode('caz', 'normal'))
        cell2_caz_reduce_vct = np.array(other.encode('caz', 'reduce'))
        cell2_caz_vct = np.concatenate((cell2_caz_normal_vct, cell2_caz_reduce_vct))

        # Compute the jackard distance
        jk_dist = np.sum(cell1_path_vct * cell2_path_vct) + np.sum(cell1_caz_vct * cell2_caz_vct)
        total_hamming_dist = np.sum(cell1_caz_vct != cell2_caz_vct) + np.sum(cell1_path_vct != cell2_path_vct)
        return total_hamming_dist / (total_hamming_dist + jk_dist)


    def encode_adj(self):
        matrices = []
        ops = []
        true_num_vertices = NUM_VERTICES + 3
        for cell in self.arch:
            matrix = np.zeros((true_num_vertices, true_num_vertices))
            op_list = []
            for i, edge in enumerate(cell):
                dest = i//2 + 2
                matrix[edge[0]][dest] = 1
                op_list.append(edge[1])
            for i in range(2, 6):
                matrix[i][-1] = 1
            matrices.append(matrix)
            ops.append(op_list)

        encoding = [*matrices[0].flatten(), *ops[0], *matrices[1].flatten(), *ops[1]]
        return np.array(encoding)

    def get_neighborhood(self,
                         nasbench, 
                         mutate_encoding='adj',
                         cutoff=None,
                         index_hash=None, 
                         shuffle=True,
                         rnd_generator=None):
        if mutate_encoding != 'adj':
            print('{} is not yet implemented as a neighborhood for nasbench301'.format(mutate_encoding))
            raise NotImplementedError()

        op_nbhd = []
        edge_nbhd = []

        for i, cell in enumerate(self.arch):
            for j, pair in enumerate(cell):

                # mutate the op
                available = [op for op in range(len(OPS)) if op != pair[1]]
                for op in available:
                    new_arch = self.make_mutable()
                    new_arch[i][j][1] = op
                    op_nbhd.append({'arch': new_arch})

                # mutate the edge
                other = j + 1 - 2 * (j % 2)
                available = [edge for edge in range(j//2+2) \
                             if edge not in [cell[other][0], pair[0]]] 

                for edge in available:
                    new_arch = self.make_mutable()
                    new_arch[i][j][0] = edge
                    edge_nbhd.append({'arch': new_arch})

        if shuffle:
            rnd_generator.shuffle(edge_nbhd)
            rnd_generator.shuffle(op_nbhd)

        # 112 in edge nbhd, 24 in op nbhd
        # alternate one edge nbr per 4 op nbrs
        nbrs = []
        op_idx = 0
        for i in range(len(edge_nbhd)):
            nbrs.append(edge_nbhd[i])
            for j in range(4):
                nbrs.append(op_nbhd[op_idx])
                op_idx += 1
        nbrs = [*nbrs, *op_nbhd[op_idx:]]

        return nbrs

    def get_num_params(self, nasbench):
        # todo: add this method
        return 100
