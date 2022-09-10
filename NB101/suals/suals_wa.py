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
        SuALS-WA
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
        self.delta = 0.7

        # SADE related variables
        self.pop_size = pop_size
        self.network = dict()
        self.solutionList = dict()
        self.setAllEstimated = set()  # Stores the all estimated solution during one run
        self.setAllTrueFitness = set()
        self.setIncumbents = set()
        self.budget = budget
        self.K = 1.4
        self.T = 3
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
        ##############
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
            cell.reliability = 1
            cell.upperLimit = 1
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

    def getEstimatedFitness(self, soln, network, solutionList, nonSameSolutions):
        ri_distance = 0
        total_ri_accuracy = 0
        total_ri_distance = 0
        for n in network[soln.solNo].keys():
            if solutionList[n] in nonSameSolutions: continue
            ri_distance = solutionList[n].reliability * np.exp(-network[soln.solNo][n])
            total_ri_accuracy += (ri_distance * solutionList[n].fitness)
            total_ri_distance += ri_distance

        return total_ri_accuracy / total_ri_distance

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

    # Çözümü estimated olarak işaretle
    def setSolEstimated(self, soln, setEstimated, setTrue):
        if soln.reliability is None:
            soln.reliability = 0.5
        soln.fitnessType = "ESTIMATED"
        setEstimated.add(soln)
        setTrue -= {soln}

    # Çözümü true fitness olarak işaretle
    def setSolTrueFitness(self, soln, setTrue, setEstimated):
        if soln.reliability is None:
            soln.reliability = 1
        soln.fitnessType = "ACTUAL"
        soln.upperLimit = 1
        setTrue.add(soln)
        setEstimated -= {soln}

    def updateUpperLimit(self, soln, network):
        if soln.fitnessType == "ACTUAL":
            soln.upperLimit = 1
        elif soln.fitnessType == "ESTIMATED":
            soln.upperLimit = min(network[soln.solNo].values())

    # Algorithm 3
    def updateNetwork(self, edge, solutionList, network):

        node1, node2, distance = solutionList[edge[0]], solutionList[edge[1]], edge[2]

        self.network.setdefault(node1.solNo, dict())
        self.network.setdefault(node2.solNo, dict())

        # edge {v, w} is added to the network only if v or w not equal T
        if (node1.fitnessType == "ACTUAL") and (node2.fitnessType == "ACTUAL"):
            return network
        else:
            minUpperLimit = self.K * min(node1.upperLimit, node2.upperLimit)
            if distance <= minUpperLimit:
                # Add edge {v, w}
                network[node1.solNo][node2.solNo] = distance
                network[node2.solNo][node1.solNo] = distance

                # After determining the node types, dUL(v) and dUL(w) are updated
                self.updateUpperLimit(node1, network)
                self.updateUpperLimit(node2, network)

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

    def calculateReliabilityVals(self):
        # Converge edilecek çözümler

        isConverged = False
        reliabilityHistory = dict()  # Stop Condition 2 için gerekli
        convergenceList = copy.copy(self.setAllEstimated)
        while isConverged == False:

            reliabilityDiffList = []  # Stop Condition 3 için gerekli
            nonConvergedSols = list()
            newReliabilityList = dict()

            for sol in convergenceList:
                solNo = sol.solNo
                prevR = sol.reliability
                r = sum(self.solutionList[n].reliability * np.exp(-self.network[solNo][n]) for n in
                        self.network[solNo].keys())
                r /= len(self.network[solNo])
                newReliabilityList[solNo] = r
                diff = np.round(np.abs(r - prevR), 7)
                reliabilityHistory.setdefault(solNo, []).append(r)
                reliabilityDiffList.append(diff)

                # Stop Conditions
                if diff == 0:  # Stop Condition - 1
                    continue
                else:
                    if len(reliabilityHistory[solNo]) >= 4:
                        _diff = np.round(np.abs(r - reliabilityHistory[solNo][-4]), 7)
                        if _diff <= 0.0001:  # Stop Condition - 2
                            continue
                nonConvergedSols.append(sol)

            convergenceList = nonConvergedSols
            # İlgili iterasyon bittikten sonra reliability değerleri güncellenecek
            for solNo, newR in newReliabilityList.items():
                self.solutionList[solNo].reliability = newR

            # Stop Condition - 3
            if len(reliabilityDiffList) == 0 or max(reliabilityDiffList) <= 0.001:
                isConverged = True

    def calculateEstimateForSoln(self, network, solutionList):

        isConverged = False
        fitnessHistory = dict()
        convergenceList = copy.copy(self.setAllEstimated)

        for sol in self.setAllEstimated:
            if sol.isFeasible and sol.fitnessType == "ESTIMATED" and sol.fitness == 0:
                sol.fitness = 0.5  # İlk fitness ataması

        while isConverged == False:

            fitnessDiffList = []
            newFitnessList = dict()
            nonConvergedSols = list()

            for sol in convergenceList:

                solNo = sol.solNo
                sol.fitnessType = "ESTIMATED"
                prevFitness = sol.fitness  # Accuracy
                newFitness = self.getEstimatedFitness(sol, network, solutionList, [])  # Accuracy
                newFitnessList[solNo] = newFitness
                diff = np.round(np.abs(newFitness - prevFitness), 7)
                fitnessHistory.setdefault(solNo, []).append(newFitness)
                fitnessDiffList.append(diff)

                # Stop Conditions
                if diff == 0:  # Stop Condition - 1
                    continue
                else:
                    if len(fitnessHistory[solNo]) >= 4:
                        _diff = np.round(np.abs(newFitness - fitnessHistory[solNo][-4]), 7)
                        if _diff <= 0.0001:  # Stop Condition - 2
                            continue
                nonConvergedSols.append(sol)

            convergenceList = nonConvergedSols
            # İlgili iterasyon bittikten sonra reliability değerleri güncellenecek
            for solNo, newFitness in newFitnessList.items():
                self.solutionList[solNo].fitness = np.round(newFitness, 7)  # Error

            # Stop Condition - 3
            if len(fitnessDiffList) == 0 or max(fitnessDiffList) <= 0.001:
                isConverged = True

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
                self.setAllEstimated -= {soln}
                setEstimated -= {soln}
                removed_sols.add(soln)
        
        solutions = list(set(solutions) - removed_sols)
        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrEstimatedSoln += len(setEstimated)

        # Reliability Calculation
        self.calculateReliabilityVals()
        
        # // Steps 7 and 8 in Figure 1
        self.calculateEstimateForSoln(self.network, self.solutionList)
    
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

        # Termination Condition
        if sum(self.runtime) > 8e4:
            return True

        return False
    
    # Return correlation between estimated solution fitness and it's true fitness values
    def calcCorrelation(self):
        trueFitnessList = []
        estimatedFitnessList = []
        self.nbrActualSoln = 0
        self.nbrEstimatedSoln = 0

        for k in self.network.keys():
            soln = self.solutionList[k]
            if soln.fitnessType == "ESTIMATED" and soln.fitness != 0:
                estimatedFitnessList.append(soln.fitness)
                if soln.trueFitness is None:
                    fitness, cost = self.f_objective(soln, budget=108, addResultFile=False)
                    soln.trueFitness = fitness
                trueFitnessList.append(soln.trueFitness)
                self.nbrEstimatedSoln += 1

            if soln.fitnessType == "ACTUAL" and soln.fitness != 0:
                self.nbrActualSoln += 1
        try:
            absoluteError = np.mean(np.absolute(np.array(estimatedFitnessList) - np.array(trueFitnessList)))
            pearson, _ = stats.pearsonr(estimatedFitnessList, trueFitnessList)
            spearman, _ = stats.spearmanr(estimatedFitnessList, trueFitnessList)
            kendalltau, _ = stats.kendalltau(estimatedFitnessList, trueFitnessList)
        except:
            return 0, 0, 0, 0

        return pearson, spearman, kendalltau, absoluteError

    def next_step(self, incumbent):
        neighbors = incumbent.get_neighborhood(self.db.dataset, rnd_generator=self.neighbor_rnd)
        return set(), set(), [], neighbors

    def local_search(self):
        print("Initializing Phase...")
        incumbent = self.init_phase(sample_size=10)
        self.setIncumbents.add(incumbent)
        neighbors = incumbent.get_neighborhood(self.db.dataset, rnd_generator=self.neighbor_rnd)

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
                     #print("NEIGHBORS IS EMPTY")
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
                    # Termination Condition 
                    if self.termination_check():
                        break
                else:
                    #print("SET ESTIMATED IS EMPTY")
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

        self.reset()
        self.local_search()

        return np.array(self.runtime)
