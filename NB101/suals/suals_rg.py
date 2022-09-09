import copy
import pickle
import nasbench
import numpy as np
import ConfigSpace
from scipy import stats
from nas_bench_101.cell_101 import Cell101
from nas_bench_101.distances import *
from collections import deque
from nasbench.lib import graph_util
from tabular_benchmarks import NASCifar10A


class LocalSearch():
    '''
        SuALS-RG
    '''

    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None,
                 budget=None, configspace=True, **kwargs):

        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # LS Parameters
        self.max_true = 3 # max no solns per iter to directly query true accuracy
        self.max_true_from_est = 5 # max no solns with estimated acc to  be selected for querying true acc per iter
        self.num_est = 10 # no solns with estimated accuracy per iter
        self.n_min = 40
        self.delta = 0.7

        # SADE related variables
        self.pop_size = pop_size
        self.network = dict()
        self.solutionList = dict()
        self.setAllEstimated = set()  # Stores the all estimated solution during one run
        self.setAllTrueFitness = set()
        self.setIncumbents = set()
        self.budget = budget
        self.T = 3 # A*'a en yakın T çözümü isolated kümesinden trueFitness kümesine al
        self.threshold = 0.1
        self.db = kwargs['b']
        self.nbrEstimatedSoln = 0
        self.nbrActualSoln = 0

        # Miscellaneous
        self.configspace = configspace
        self.NUM_VERTICES = 7
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Global trackers
        self.inc_score = 0
        self.inc_config = None
        self.population = None
        self.fitness = None

    def save_data(self, data, filename):
        fh = open(f'{self.output_path}/{filename}.pkl', 'wb')
        pickle.dump(data, fh)
        fh.close()

    def reset(self):
        self.inc_score = 0
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.network = dict()
        self.solutionList = dict()
        self.setAllEstimated = set()
        self.setAllTrueFitness = set()
        self.setIncumbents = set()
        self.runtime = []
        self.initRndNumberGenerators()
    
    def initRndNumberGenerators(self):
        # Random Number Generators
        self.init_rnd = np.random.RandomState(self.seed)
        self.neighbor_rnd = np.random.RandomState(self.seed)
        self.converter_rnd = np.random.RandomState(self.seed)

    def vector_to_cell(self, chromosome):
        VERTICES = 7
        NBR_OPS = 5

        # converts [0, 1] vector to matrix
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        matrix[idx] = chromosome[:-NBR_OPS] > 0.5

        # converts [0, 1] vector to a ConfigSpace object
        config = self.vector_to_configspace(chromosome)
        ops = [config[f"op_node_{key}"] for key in range(5)]
        ops = ['input'] + list(ops) + ['output']

        # Create Cell Object
        cell = Cell101(chromosome, matrix, ops)
        cell.solNo = self.solNo
        cell.config = config
        cell.isFeasible = self.db.feasibilityCheck(cell)

        if cell.isFeasible == False:
            cell.fitness = 0
            cell.cost = 0
        
        self.solNo = self.solNo + 1
        return cell

    def vector_to_configspace(self, vector):
        '''Converts numpy array to ConfigSpace object

        Works when self.cs is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        new_config = self.cs.sample_configuration()
        for i, hyper in enumerate(self.cs.get_hyperparameters()):
            if 'op' in hyper.name:
                i = int(hyper.name.split("_")[2]) + 21
            else:
                i = int(hyper.name.split("_")[1])
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = np.round(param_value).astype(int)  # converting to discrete (int)
            new_config[hyper.name] = param_value
        return new_config

    def checkSolution(self, cell):
        for solNo, solRef in self.solutionList.items():
            D = jackard_distance_caz(cell, solRef)
            if D == 0:
                return True, solNo

        return False, cell.solNo

    def f_objective(self, cell, budget=None, addResultFile=None):

        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            fitness, cost = self.f(cell, budget=budget, addResultFile=addResultFile)
        else:
            fitness, cost = self.f(cell)
        return fitness, cost

    def init_random_samples(self, sample_size):
        samples = []
        for idx in range(sample_size):
            chromosome = self.init_rnd.uniform(low=0.0, high=1.0, size=self.dimensions)
            cell = self.vector_to_cell(chromosome)
            cost, fitness = self.getActualForSoln(cell)

            self.solutionList[cell.solNo] = cell

            samples.append(cell)

        return np.array(samples)


    def init_phase(self, sample_size):
        samples = self.init_random_samples(sample_size)

        setTrue = copy.deepcopy(samples)
        # Create Candidate Edge List
        candidateEdges = set()  # the set of candidate edges
        for soln in samples:
            candidateEdges = self.getSolutionCandidateEdges(soln, self.solutionList, self.delta, candidateEdges)

        # The edges in C are sorted in increasing distance
        candidateEdges = sorted(candidateEdges, key=lambda x: x[2])

        # Update Network
        for edge in candidateEdges:
            self.network = self.updateNetwork(edge, self.solutionList, self.network)
        
        
        # Isolated Solutions
        for soln in samples:
            self.network.setdefault(soln.solNo, dict())
            if soln.fitnessType == 'ESTIMATED' and len(self.network[soln.solNo]) < 1: # Isolated Estimated Solutions
                del self.network[soln.solNo]
                del self.solutionList[soln.solNo]
        
        self.nbrActualSoln += len(setTrue)

        for soln in samples:
            # Best Solution
            if soln.fitness >= self.inc_score:
                self.inc_score = soln.fitness
                self.inc_config = soln
        
        return self.inc_config # Return current incumbent

    def getEstimatedFitness(self, soln):
        numerator = 0
        denominator = 0
        for solnNo, distance in self.network[soln.solNo].items():
            if self.solutionList[solnNo].fitness != 0:
                numerator += ((1 - distance) * self.solutionList[solnNo].fitness)
                denominator += (1 - distance)
        
        return numerator / denominator

    def addEstimatedNeighbors(self, neighbors, solutionList):
        setEstimated = set()
        for n in neighbors:
            neighbor = solutionList[n]
            if neighbor.fitnessType == "ESTIMATED":
                setEstimated.add(neighbor)
        return setEstimated

    # Get candidate edges for given solution as parameter
    def getSolutionCandidateEdges(self, soln, solutionList, delta, candidateEdges):
        for indvNo, indv in solutionList.items():
            if indv.solNo == soln.solNo:
                continue

            # Calculate Jackard Distance between indv and soln
            distance = jackard_distance_caz(indv, soln)
            if distance < delta:
                minSolNo = min(indv.solNo, soln.solNo)
                maxSolNo = max(indv.solNo, soln.solNo)
                candidateEdges.add((minSolNo, maxSolNo, distance))

        return candidateEdges

    
    def setSolEstimated(self, soln, setEstimated, setTrue):
        if soln.reliability is None:
            soln.reliability = 0.5
        soln.fitnessType = "ESTIMATED"
        setEstimated.add(soln)
        setTrue -= {soln}

    
    def setSolTrueFitness(self, soln, setTrue, setEstimated):
        if soln.reliability is None:
            soln.reliability = 1
        soln.fitnessType = "ACTUAL"
        soln.upperLimit = 1
        setTrue.add(soln)
        setEstimated -= {soln}

    # Algorithm 5
    def calcEstimatedFitness(self, soln):

        regression_models = [self.solutionList[n] for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].regression_model is not None and self.solutionList[n].fitness != 0]

        if len(regression_models) > 0:
            soln.fitness = (sum(n.regression_model.predict([[self.network[soln.solNo][n.solNo]]])[0][0] for n in regression_models if n.regression_model is not None) / len(regression_models))
            soln.estimation_type = 'regression'
            
        return len(regression_models)

    # Algorithm 3
    def updateNetwork(self, edge, solutionList, network):

        node1, node2, distance = solutionList[edge[0]], solutionList[edge[1]], edge[2]

        self.network.setdefault(node1.solNo, dict())
        self.network.setdefault(node2.solNo, dict())     
            
        network[node1.solNo][node2.solNo] = distance
        network[node2.solNo][node1.solNo] = distance

        return self.network

    def best_solution_check(self, sol):
        if sol.fitness > self.inc_score:
            self.inc_score = sol.fitness
            self.inc_config = sol

    # Algorithm 4
    def getActualForSoln(self, soln):
        fitness, cost = self.f_objective(soln, budget=108, addResultFile=True)
        self.runtime.append(cost)
        soln.fitness = fitness
        soln.fitnessType = "ACTUAL"
        soln.cost = cost
        soln.upperLimit = 1
        self.setAllTrueFitness.add(soln)
        self.best_solution_check(soln)

        return cost, fitness
    
    def regression_model(self, soln):
        true_neighbors = [self.solutionList[n] for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].fitness != 0]
        if len(true_neighbors) >= self.n_min:
            X = [[self.network[soln.solNo][n.solNo]] for n in true_neighbors]
            y = [[n.fitness] for n in true_neighbors]
            soln.fit_regression_model(X, y)

    def get_cont_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1.0001, step=1/step_size)
        return self.converter_rnd.uniform(low=ranges[value], high=ranges[value + 1])
    
    def config_to_vector(self, matrix, ops):
        '''Converts discrete values to numpy array'''
        
        triu_indices = np.triu_indices(self.NUM_VERTICES, k = 1)
        config = np.zeros(self.dimensions, dtype='uint8')
        vector = np.zeros(self.dimensions, dtype='float32')
        max_edges = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)

        # Edges
        config[:max_edges] = matrix[triu_indices]
        for idx in range(max_edges):
            vector[idx] = self.get_cont_value(config[idx], 2)
        
        # Vertices - Ops
        config[max_edges: max_edges + self.NUM_VERTICES - 2] = ops
        for idx in range(max_edges, max_edges + self.NUM_VERTICES - 2):
            vector[idx] = self.get_cont_value(config[idx], len(self.OPS))
        
        return config, vector    

    def termination_check(self):
        if sum(self.runtime) > 8e4:
            return True
        return False

    def mark_solution(self, neighbors, nonSameSolutions, type, setTrue, setEstimated):
        while len(neighbors) > 0:
            neighbor = neighbors.pop(0)
            cell = Cell101(**neighbor)
            if self.checkSolution(cell)[0]:
                continue
            cell.solNo = self.solNo
            self.solutionList[cell.solNo] = cell
            nonSameSolutions.append(cell)
            if type == 'ACTUAL':
                self.setSolTrueFitness(cell, setTrue, setEstimated)
                self.getActualForSoln(cell)
            else:
                self.setSolEstimated(cell, setEstimated, setTrue)

            self.solNo += 1

            return self.termination_check(), cell
        
        return self.termination_check(), None

    def estimated_step(self, setEstimated, solutions):      
        # Create Candidate Edge List
        candidateEdges = set() # the set of candidate edges
        for soln in solutions:
            candidateEdges = self.getSolutionCandidateEdges(soln, self.solutionList, self.delta, candidateEdges)

        # The edges in C are sorted in increasing distance
        candidateEdges = sorted(candidateEdges, key=lambda x: x[2])

        # Update Network
        for edge in candidateEdges:
            self.network = self.updateNetwork(edge, self.solutionList, self.network)
        
        removed_sols = set()
        for soln in solutions:
            self.network.setdefault(soln.solNo, dict())
            if soln.fitnessType == 'ESTIMATED' and len(self.network[soln.solNo]) < 1: # Isolated Estimated Solutions
                del self.network[soln.solNo]
                del self.solutionList[soln.solNo]
                setEstimated -= {soln}
                removed_sols.add(soln)
                print("ISOLATED:",soln.solNo)
        
        solutions = list(set(solutions) - removed_sols)
        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrEstimatedSoln += len(setEstimated)
        
        true_neighbors = set() # set R
        try:
            for soln in solutions:
                true_neighbors |= {n for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].fitness != 0}
        except:
            print()
        # Create/Update Regression Model
        for soln in true_neighbors:
            self.regression_model(self.solutionList[soln])

        for soln in solutions:
            nbr_true_neighbors = self.calcEstimatedFitness(soln)
    
    def true_step(self, soln):
        # Create Candidate Edge List
        candidateEdges = set() # the set of candidate edges

        candidateEdges = self.getSolutionCandidateEdges(soln, self.solutionList, self.delta, candidateEdges)

        # The edges in C are sorted in increasing distance
        candidateEdges = sorted(candidateEdges, key=lambda x: x[2])

        # Update Network
        for edge in candidateEdges:
            self.network = self.updateNetwork(edge, self.solutionList, self.network)

        self.nbrActualSoln += 1
    
    def estimated_to_true(self, sol, setEstimated, setTrue):
        self.getActualForSoln(sol)
        self.setSolTrueFitness(sol, setTrue, setEstimated)
        self.nbrActualSoln += 1
        sol.reliability = 1
        setEstimated -= {sol}
        self.setAllEstimated -= {sol}
        self.regression_model(sol) # Create Regression Model

        # Termination Condition
        if sum(self.runtime) > 8e4:
            return True

        return False
    
    def next_step(self, incumbent):
        neighbors = incumbent.get_neighborhood(self.db.dataset, rnd_generator=self.neighbor_rnd, index_hash=self.hash_index)
        return set(), set(), [], neighbors

    def local_search(self):
        print("Initializing Phase...")
        incumbent = self.init_phase(sample_size=10)
        self.setIncumbents.add(incumbent)
        neighbors = incumbent.get_neighborhood(self.db.dataset, rnd_generator=self.neighbor_rnd, index_hash=self.hash_index)

        self.i = 0
        ctr_true = 0
        ctr_true_from_est = 0
        nonSameSolutions = []
        setTrue = set()
        setEstimated = set()
        print("Local Search...")

        while sum(self.runtime) < 8e4:
            if ctr_true < self.max_true:
                terminate, v_i = self.mark_solution(neighbors, nonSameSolutions, 'ACTUAL', setTrue, setEstimated)
                if terminate:
                    break

                if v_i is not None:
                    self.true_step(v_i)
                else:
                    ctr_true = self.max_true + 1
                    ctr_true_from_est = self.max_true_from_est
                 
                ctr_true += 1
            else:
                if ctr_true_from_est == 0:
                    solutions = []
                    while len(solutions) < self.num_est and len(neighbors) > 0:
                        terminate, cell = self.mark_solution(neighbors, nonSameSolutions, 'ESTIMATED', setTrue, setEstimated)
                        if cell is not None:
                            solutions.append(cell)
                    
                    if len(setEstimated) > 0:
                        self.estimated_step(setEstimated, solutions)

                if len(setEstimated) > 0:
                    ctr_true_from_est += 1
                    v_i = max(setEstimated, key=lambda x: x.fitness) # best estimated
                    self.estimated_to_true(v_i, setEstimated, setTrue)

                    # Update Regression Models of v_i ACTUAL Neighbors
                    for neighbor in self.network[v_i.solNo].keys():
                        if self.solutionList[neighbor].fitnessType == "ACTUAL":
                            self.regression_model(self.solutionList[neighbor])
                    
                    # Update Estimated Values
                    for soln in setEstimated:
                        self.calcEstimatedFitness(soln)

                    # Termination Condition 
                    if self.termination_check():
                        break
                else:
                    ctr_true_from_est = self.max_true_from_est
            
            if v_i is not None and v_i.fitness > incumbent.fitness:
                self.i += 1
                ctr_true = 0
                ctr_true_from_est = 0
                incumbent = v_i
                self.setIncumbents.add(v_i)
                setTrue, setEstimated, nonSameSolutions, neighbors = self.next_step(incumbent)
            elif ctr_true_from_est == self.max_true_from_est:
                incumbent = max(self.setAllTrueFitness - self.setIncumbents, key=lambda x: x.fitness)
                self.setIncumbents.add(incumbent)
                self.i += 1
                ctr_true = 0
                ctr_true_from_est = 0
                setTrue, setEstimated, nonSameSolutions, neighbors = self.next_step(incumbent)
                
            if sum(self.runtime) > 6e4:
                print(sum(self.runtime), self.inc_score)

    def run(self, generations=1, verbose=False, budget=None, reset=True, seed=None):
        generation = 1  # Generation Number
        self.seed = seed
        self.solNo = 0
        self.terminate = False
        self.runtime = []

        self.hash_index = None

        self.reset()
        self.local_search()

        return np.array(self.runtime)
