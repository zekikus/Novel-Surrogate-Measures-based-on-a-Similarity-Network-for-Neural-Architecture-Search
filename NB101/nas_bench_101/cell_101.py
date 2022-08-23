import copy
import numpy as np
from nasbench import api
from nas_bench_101.encodings import *
from sklearn.linear_model import LinearRegression

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
OPS_INCLUSIVE = [INPUT, OUTPUT, *OPS]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class Cell101:

    def __init__(self, chromosome, matrix, ops):
        self.solNo = None
        self.matrix = matrix
        self.ops = ops
        self.fitness = 0
        self.trueFitness = None
        self.config = None
        self.fitnessType = None
        self.nbrActualNeighbor = 0
        self.isFeasible = None
        self.cost = 0 # Training Cost
        self.chromosome = chromosome
        self.reliability = None
        self.upperLimit = 1
        self.reliabilityHistory = []
        self.fitnessHistory = []
        self.regression_model = None
        self.regression_sample_size = None
        self.neighbors = None
        self.estimation_type = None

    def get_matrix(self):
        return self.matrix

    def get_ops(self):
        return self.ops
    
    def fit_regression_model(self, X, y):
        self.regression_sample_size = len(X)
        self.regression_model = LinearRegression().fit(X, y)

    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def get_utilized(self):
        # return the sets of utilized edges and nodes
        # first, compute all paths
        n = np.shape(self.matrix)[0]
        sub_paths = []
        for j in range(0, n):
            sub_paths.append([[(0, j)]]) if self.matrix[0][j] else sub_paths.append([])
        
        # create paths sequentially
        for i in range(1, n - 1):
            for j in range(1, n):
                if self.matrix[i][j]:
                    for sub_path in sub_paths[i]:
                        sub_paths[j].append([*sub_path, (i, j)])
        paths = sub_paths[-1]

        utilized_edges = []
        for path in paths:
            for edge in path:
                if edge not in utilized_edges:
                    utilized_edges.append(edge)

        utilized_nodes = []
        for i in range(NUM_VERTICES):
            for edge in utilized_edges:
                if i in edge and i not in utilized_nodes:
                    utilized_nodes.append(i)

        return utilized_edges, utilized_nodes

    def num_edges_and_vertices(self):
        # return the true number of edges and vertices
        edges, nodes = self.get_utilized()
        return len(edges), len(nodes)

    def is_valid_vertex(self, vertex):
        edges, nodes = self.get_utilized()
        return (vertex in nodes)

    def is_valid_edge(self, edge):
        edges, nodes = self.get_utilized()
        return (edge in edges)

    @classmethod
    def convert_to_cell(cls, arch):
        matrix, ops = arch['matrix'], arch['ops']

        if len(matrix) < 7:
            # the nasbench spec can have an adjacency matrix of n x n for n<7, 
            # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
            # so this method will add a blank row/column

            new_matrix = np.zeros((7, 7), dtype='int8')
            new_ops = []
            n = matrix.shape[0]
            for i in range(7):
                for j in range(7):
                    if j < n - 1 and i < n:
                        new_matrix[i][j] = matrix[i][j]
                    elif j == n - 1 and i < n:
                        new_matrix[i][-1] = matrix[i][j]

            for i in range(7):
                if i < n - 1:
                    new_ops.append(ops[i])
                elif i < 6:
                    new_ops.append('conv3x3-bn-relu')
                else:
                    new_ops.append('output')
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

        else:
            return {
                'matrix': matrix,
                'ops': ops
            }

    def encode(self, predictor_encoding, nasbench=None, deterministic=True, cutoff=None):

        if predictor_encoding == 'path':
            return encode_paths(self.get_path_indices())
        elif predictor_encoding == 'caz':
            return encode_caz(self.matrix, self.ops)
        else:
            print('{} is an invalid predictor encoding'.format(predictor_encoding))
            raise NotImplementedError()

    def get_num_params(self, nasbench):
        """
        Return trainable parameters of Cell_101
        """
        return nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['trainable_parameters']

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        paths = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)

        path_indices.sort()
        return tuple(path_indices)
    
    def get_neighborhood(self,
                         nasbench, 
                         mutate_encoding='adj',
                         cutoff=None,
                         index_hash=None, 
                         rnd_generator=None):
        if mutate_encoding == 'adj':
            return self.adj_neighborhood(nasbench,
                                         rnd_generator=rnd_generator)
        elif mutate_encoding in ['path', 'trunc_path']:
            if 'trunc' in mutate_encoding and not cutoff:
                cutoff = 40
            elif 'trunc' not in mutate_encoding:
                cutoff = None
            return self.path_neighborhood(nasbench,
                                          cutoff=cutoff,
                                          index_hash=index_hash,
                                          rnd_generator=rnd_generator)
        else:
            print('{} is an invalid neighborhood encoding'.format(mutate_encoding))
            raise NotImplementedError()
    
    def adj_neighborhood(self, nasbench, rnd_generator=None):

        """
        if self.neighbors is not None: #YENİ EKLENDİ
            return self.neighbors #YENİ EKLENDİ
        """
        nbhd = []
        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            #if self.is_valid_vertex(vertex):
            available = [op for op in OPS if op != self.ops[vertex]]
            for op in available:
                new_matrix = copy.deepcopy(self.matrix)
                new_ops = copy.deepcopy(self.ops)
                new_ops[vertex] = op
                new_arch = {'matrix':new_matrix, 'ops':new_ops, 'chromosome': None}
                nbhd.append(new_arch)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src+1, NUM_VERTICES):
                new_matrix = copy.deepcopy(self.matrix)
                new_ops = copy.deepcopy(self.ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_arch = {'matrix':new_matrix, 'ops':new_ops, 'chromosome': None}
            
                #if self.matrix[src][dst] and self.is_valid_edge((src, dst)):
                spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                    #if nasbench.is_valid(spec):                            
                nbhd.append(new_arch)  

                """
                if not self.matrix[src][dst] and Cell101(**new_arch).is_valid_edge((src, dst)):
                    spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if nasbench.is_valid(spec):                            
                        nbhd.append(new_arch)
                """

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src+1, NUM_VERTICES):
                new_matrix = copy.deepcopy(self.matrix)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]

                # add op neighbors
                for vertex in range(1, OP_SPOTS + 1):
                    #if self.is_valid_vertex(vertex):
                    available = [op for op in OPS if op != self.ops[vertex]]
                    for op in available:
                        new_ops = copy.deepcopy(self.ops)
                        new_ops[vertex] = op
                        new_arch = {'matrix':new_matrix, 'ops':new_ops, 'chromosome': None}
                        #if self.matrix[src][dst] and self.is_valid_edge((src, dst)):
                        spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                        #if nasbench.is_valid(spec):                           
                        nbhd.append(new_arch)  

                        """
                        if not self.matrix[src][dst] and Cell101(**new_arch).is_valid_edge((src, dst)):
                            spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                            if nasbench.is_valid(spec):                            
                                nbhd.append(new_arch)
                        """

        rnd_generator.shuffle(nbhd)
        self.neighbors = nbhd #YENİ EKLENDİ
        return nbhd
    
    """
    def adj_neighborhood(self, nasbench, rnd_generator=None):
        nbhd = []
        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            available = [op for op in OPS if op != self.ops[vertex]]
            for op in available:
                new_matrix = copy.deepcopy(self.matrix)
                new_ops = copy.deepcopy(self.ops)
                new_ops[vertex] = op
                new_arch = {'matrix':new_matrix, 'ops':new_ops, 'chromosome': None}
                nbhd.append(new_arch)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src+1, NUM_VERTICES):
                new_matrix = copy.deepcopy(self.matrix)
                new_ops = copy.deepcopy(self.ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_arch = {'matrix':new_matrix, 'ops':new_ops, 'chromosome': None}
  
                nbhd.append(new_arch)  

        rnd_generator.shuffle(nbhd)
        return nbhd
    """
    def path_neighborhood(self, 
                          nasbench,
                          cutoff,
                          index_hash,
                          rnd_generator):
        """
        For NAS encodings experiments, some of the path-based encodings currently require a
        hash map from path indices to cell architectuers. We have created a pickle file which
        contains the hash map, located at 
        https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing
        """

        nbhd = []
        path_indices = self.get_path_indices()
        total_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])

        if cutoff:
            cutoff_value = cutoff
        else:
            cutoff_value = total_paths

        new_sets = []
        path_indices_cutoff = [path for path in path_indices if path < cutoff_value]

        # remove paths
        for path in path_indices_cutoff:
            new_path_indices = [p for p in path_indices if p != path]
            new_sets.append(new_path_indices)

        # add paths
        other_paths = [path for path in range(cutoff_value) if path not in path_indices]
        for path in other_paths:
            new_path_indices = [*path_indices, path]
            new_sets.append(new_path_indices)

        for new_path_indices in new_sets:
            new_tuple = tuple(new_path_indices)
            if new_tuple in index_hash:

                spec = index_hash[new_tuple]
                matrix = spec['matrix']
                ops = spec['ops']
                model_spec = api.ModelSpec(matrix=matrix, ops=ops)
                #if nasbench.is_valid(model_spec):
                spec['chromosome'] = None                            
                nbhd.append(spec)


        rnd_generator.shuffle(nbhd)
        return nbhd
