import copy
import pickle
import numpy as np
import ConfigSpace
from scipy import stats
from nas_bench_101.cell_101 import Cell101
from nas_bench_101.distances import *
from collections import deque
from nasbench.lib import graph_util


class SADEBase():
    '''
        SuADE-WA
    '''

    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=None,
                 mutation_factor=None, crossover_prob=None, strategy=None, budget=None,
                 configspace=True, boundary_fix_type='random', **kwargs):

        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # SADE related variables
        self.pop_size = pop_size
        self.network = dict()
        self.solutionList = dict()
        self.setAllEstimated = set()  # Stores the all estimated solution during one run
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.budget = budget
        self.fix_type = boundary_fix_type
        self.K = 1
        self.T = 3
        self.delta = 0.5
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

    def init_population(self, pop_size=None):
        print("POP SIZE:", pop_size)
        chromosomes = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        population = []
        for i in range(pop_size):
            chromosome = copy.copy(chromosomes[i])
            cell = self.vector_to_cell(chromosome)
            population.append(cell)
        
        del chromosomes # Clear Memory
        return np.array(population)

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

    def sample_population(self, size=3, alt_pop=None):
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population
        '''
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = self.sample_pop_rnd.choice(np.arange(len(self.population)), size, replace=False)
                return self.population[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                selection = self.sample_pop_rnd.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return alt_pop[selection]
        else:
            selection = self.sample_pop_rnd.choice(np.arange(len(self.population)), size, replace=False)
            return self.population[selection]

    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        if fix_type == 'random', the values are replaced with a random sampling from (0,1)
        if fix_type == 'clip', the values are clipped to the closest limit from {0, 1}

        Parameters
        ----------
        vector : array

        Returns
        -------
        array
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.fix_type == 'random':
            vector[violations] = self.boundary_rnd.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

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

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def mutation(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def crossover(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def evolve(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class SADE(SADEBase):
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 budget=None, encoding=False, dim_map=None, **kwargs):

        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, budget=budget, **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.encoding = encoding
        self.dim_map = dim_map

    def initRndNumberGenerators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.boundary_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.drago_rnd = np.random.RandomState(self.seed)
        self.type_decision_rnd = np.random.RandomState(self.seed)

    def reset(self, seed):
        super().reset()
        self.runtime = []
        self.setAllEstimated = set()
        self.initRndNumberGenerators()

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

    def init_eval_pop(self, budget=None, eval=True):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_population(self.pop_size)

        setTrue = set()
        setEstimated = set()
        runtime = []
        nonSameSolutions = []
        totalTime = sum(self.runtime) 
        generationCost = 0 

        for i in range(self.pop_size):
            cell = self.population[i]
            if cell.isFeasible:
                # Isomorphism Check
                isSameSolution, solNo = self.checkSolution(cell)
                if isSameSolution:
                    print("SAME SOLUTION:")
                    self.population[i] = self.solutionList[solNo]
                    cell = self.population[i]
                else:
                    self.solutionList[cell.solNo] = cell
                    nonSameSolutions.append(cell)
                ###################

        nonSameSolutions = np.array(nonSameSolutions)

        # Create Candidate Edge List
        candidateEdges = set()  # the set of candidate edges
        for soln in nonSameSolutions:
            candidateEdges = self.getSolutionCandidateEdges(soln, self.solutionList, self.delta, candidateEdges)

        # The edges in C are sorted in increasing distance
        candidateEdges = sorted(candidateEdges, key=lambda x: x[2])

        # Update Network
        for edge in candidateEdges:
            self.network, setEstimated, setTrue = self.updateNetwork(edge, self.solutionList, nonSameSolutions,
                                                                     self.network, setTrue, setEstimated)
        # Isolated Solutions
        for soln in nonSameSolutions:
            self.network.setdefault(soln.solNo, dict())
            if len(self.network[soln.solNo]) < 1:
                self.setSolTrueFitness(soln, setTrue)

        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrActualSoln += len(setTrue)
        self.nbrEstimatedSoln += len(setEstimated)

        # Reliability Calculation
        self.calculateReliabilityVals()

        # // Step 6 in Figure 1
        for soln in setTrue:
            cost, fitness = self.getActualForSoln(soln, self.solutionList)
            runtime.append(cost)
            generationCost += cost

            # Best Solution
            if fitness > self.inc_score:
                self.inc_score = fitness
                self.inc_config = soln

            # Termination Condition
            if totalTime + generationCost > 8e4:
                self.terminate = True
                return runtime

        # // Steps 7 and 8 in Figure 1
        self.calculateEstimateForSoln(self.network, self.solutionList)

        return runtime

    def mutation_currenttobest1(self, current, best, r1, r2):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff1 = best.chromosome - current
        diff2 = r1.chromosome - r2.chromosome
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current, best):
        '''Performs DE mutation
        '''
        r1, r2 = self.sample_population(size=2, alt_pop=None)
        mutant = self.mutation_currenttobest1(current, best, r1, r2)

        return self.boundary_check(mutant)

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)

        return offspring

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

    # Algorithm 2 -  Başlatma fazı için
    def chooseInitActualEstimatedSolns(self, network, solutionList, nonSameSolutions, setEstimated, setTrue):

        for i, soln in enumerate(nonSameSolutions):
            if soln.isFeasible == False or soln.fitnessType != None: continue
            if soln.fitnessType == "ACTUAL": continue

            hasActualNeighbor = False
            for n in network[soln.solNo].keys():
                neighbor = solutionList[n]  # Solution Reference

                if neighbor.fitnessType == "ACTUAL":
                    hasActualNeighbor = True
                    soln.fitnessType = "ESTIMATED"
                    if soln.reliability == None:
                        soln.reliability = 0.5
                    setEstimated.add(soln)
                    break

            if hasActualNeighbor == False:
                soln.fitnessType = "ACTUAL"
                soln.upperLimit = 1
                if soln.reliability == None:
                    soln.reliability = 1
                setTrue.add(soln)
                for n in network[soln.solNo].keys():
                    neighbor = solutionList[n]
                    if neighbor.fitnessType == "ESTIMATED":
                        setEstimated.add(neighbor)

        return setEstimated, setTrue

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

    
    def setSolEstimated(self, soln, setEstimated):
        if soln.reliability is None:
            soln.reliability = 0.5
        soln.fitnessType = "ESTIMATED"
        setEstimated.add(soln)

   
    def setSolTrueFitness(self, soln, setTrue):
        if soln.reliability is None:
            soln.reliability = 1
        soln.fitnessType = "ACTUAL"
        soln.upperLimit = 1
        setTrue.add(soln)

    def updateUpperLimit(self, soln, network):
        if soln.fitnessType == "ACTUAL":
            soln.upperLimit = 1
        elif soln.fitnessType == "ESTIMATED":
            soln.upperLimit = min(network[soln.solNo].values())

    # Algorithm 3
    def updateNetwork(self, edge, solutionList, nonSameSolutions, network, setTrue, setEstimated):
        v = None
        w = None
        node1, node2, distance = solutionList[edge[0]], solutionList[edge[1]], edge[2]

        self.network.setdefault(node1.solNo, dict())
        self.network.setdefault(node2.solNo, dict())

        # edge {v, w} is added to the network only if v or w not equal T
        if (node1.fitnessType == "ACTUAL") and (node2.fitnessType == "ACTUAL"):
            return network, setEstimated, setTrue
        else:
            minUpperLimit = self.K * min(node1.upperLimit, node2.upperLimit)
            if distance < minUpperLimit:
                # Add edge {v, w}
                network[node1.solNo][node2.solNo] = distance
                network[node2.solNo][node1.solNo] = distance

                # Type Decision
                if (node1.fitnessType is None) and (node1 in nonSameSolutions):
                    v = node1
                    w = node2
                elif (node2.fitnessType is None) and (node2 in nonSameSolutions):
                    v = node2
                    w = node1

                # assuming v is in N, type decision must be made for v;
                if (v is not None):
                    # If w ∈ T, v is added to E;
                    if w.fitnessType == "ACTUAL":
                        self.setSolEstimated(v, setEstimated)
                    # If w ∈ N, one of v or w is randomly assigned to T and the other is added to E
                    elif w in nonSameSolutions:
                        solutions = [w, v]
                        self.type_decision_rnd.shuffle(solutions)
                        # Add True Fitness List
                        setEstimated |= set([solutionList[n] for n in network[solutions[0].solNo].keys() if solutionList[n].fitnessType == "ESTIMATED"])
                        self.setSolTrueFitness(solutions[0], setTrue)
                        # Add Estimated List
                        self.setSolEstimated(solutions[1], setEstimated)
                    # If w ∈ E, then v is added to T if ˜A(w) > α, and added to E, otherwise.
                    elif w.fitnessType == "ESTIMATED":
                        fitness = self.getEstimatedFitness(w, network, solutionList, nonSameSolutions)
                        if fitness > max(self.inc_score - self.threshold, 0.6):  # Threshold Check
                            self.setSolTrueFitness(v, setTrue)
                            v.reliability = 1
                            # Add neighbors of v to the setEstimated
                            neighbors = [solutionList[n] for n in network[v.solNo].keys() if solutionList[n].fitnessType == "ESTIMATED"]
                            setEstimated |= set(neighbors)
                        else:
                            self.setSolEstimated(v, setEstimated)

                # After determining the node types, dUL(v) and dUL(w) are updated
                self.updateUpperLimit(node1, network)
                self.updateUpperLimit(node2, network)

        return self.network, setEstimated, setTrue

    def calculateReliabilityVals(self):
        # Convergence

        isConverged = False
        reliabilityHistory = dict()  # Stop Condition 2
        convergenceList = copy.copy(self.setAllEstimated)
        while isConverged == False:

            reliabilityDiffList = []  # Stop Condition 3
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
            
            for solNo, newR in newReliabilityList.items():
                self.solutionList[solNo].reliability = newR

            # Stop Condition - 3
            if len(reliabilityDiffList) == 0 or max(reliabilityDiffList) <= 0.001:
                isConverged = True

    # Algorithm 4
    def getActualForSoln(self, soln, solutionList):
        fitness, cost = self.f_objective(soln, budget=108, addResultFile=True)
        soln.fitness = fitness
        soln.fitnessType = "ACTUAL"
        soln.cost = cost
        soln.upperLimit = 1

        return cost, fitness

    # Algorithm 5
    def calculateEstimateForSoln(self, network, solutionList):

        isConverged = False
        fitnessHistory = dict()
        convergenceList = copy.copy(self.setAllEstimated)

        for sol in self.setAllEstimated:
            if sol.isFeasible and sol.fitnessType == "ESTIMATED" and sol.fitness == 0:
                sol.fitness = 0.5  # Initial fitness assignment

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
            
            for solNo, newFitness in newFitnessList.items():
                self.solutionList[solNo].fitness = np.round(newFitness, 7)  # Error

            # Stop Condition - 3
            if len(fitnessDiffList) == 0 or max(fitnessDiffList) <= 0.001:
                isConverged = True

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

    def evolve_generation(self, generation, budget=None, best=None, alt_pop=None):
        '''
        Performs a complete SADE evolution
        '''

        trials = []
        nonSameSolutions = []
        Pnext = []
        runtime = []
        setTrue = set()
        setEstimated = set()
        totalTime = sum(self.runtime)  
        generationCost = 0 

        generationBest = max(self.population, key=lambda x: x.fitness)
        
        for j in range(self.pop_size):
            target = self.population[j].chromosome
            mutant = copy.deepcopy(target)
            mutant = self.mutation(target, generationBest)
            mutant = self.crossover(target, mutant)
            cell = self.vector_to_cell(mutant)

            if cell.isFeasible:
                # Isomorphism Check
                isSameSolution, solNo = self.checkSolution(cell)
                if isSameSolution:
                    print("SAME SOLUTION")
                    cell = self.solutionList[solNo]
                else:
                    self.solutionList[cell.solNo] = cell
                    nonSameSolutions.append(cell)

            trials.append(cell)
            del mutant  # Free Memory

        trials = np.array(trials)
        nonSameSolutions = np.array(nonSameSolutions)

        # Create Candidate Edge List
        candidateEdges = set() # the set of candidate edges
        for soln in nonSameSolutions:
            candidateEdges = self.getSolutionCandidateEdges(soln, self.solutionList, self.delta, candidateEdges)

        # The edges in C are sorted in increasing distance
        candidateEdges = sorted(candidateEdges, key=lambda x: x[2])

        # Update Network
        for edge in candidateEdges:
            self.network, setEstimated, setTrue = self.updateNetwork(edge, self.solutionList, nonSameSolutions, self.network, setTrue, setEstimated)

        candidateTrueFitness = []
        # Candidate Isolated Solutions
        for soln in nonSameSolutions:
            self.network.setdefault(soln.solNo, dict())
            if len(self.network[soln.solNo]) < 1:
                distance = jackard_distance_caz(self.inc_config, soln)
                candidateTrueFitness.append((soln, distance))

        # Sort candidate isolated solution list
        candidateTrueFitness = sorted(candidateTrueFitness, key=lambda x: x[1])

        # Add the closest solutions to the true Fitness set
        for soln, d in candidateTrueFitness[:self.T]:
            self.setSolTrueFitness(soln, setTrue)

        for soln, d in candidateTrueFitness[self.T:]:
            del self.network[soln.solNo]
            del self.solutionList[soln.solNo]
            soln.fitness = 0

        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrActualSoln += len(setTrue)
        self.nbrEstimatedSoln += len(setEstimated)

        # Reliability Calculation
        self.calculateReliabilityVals()

        # // Step 6 in Figure 1
        for soln in setTrue:
            cost, fitness = self.getActualForSoln(soln, self.solutionList)
            runtime.append(cost)
            generationCost += cost

            # Best Solution
            if fitness > self.inc_score:
                self.inc_score = fitness
                self.inc_config = soln

            # Termination Condition
            if totalTime + generationCost > 8e4:
                self.terminate = True
                return runtime

        # // Steps 7 and 8 in Figure 1
        self.calculateEstimateForSoln(self.network, self.solutionList)

        # // Step 9 in Figure 1
        for j in range(self.pop_size):
            target = self.population[j]
            mutant = trials[j]

            if mutant.fitness >= target.fitness:
                Pnext.append(mutant)
            else:
                Pnext.append(target)

        self.population = np.array(Pnext)

        return runtime

    def run(self, generations=1, verbose=False, budget=None, reset=True, seed=None):
        generation = 1  # Generation Number
        self.seed = seed
        self.solNo = 0
        self.terminate = False

        # checking if a run exists
        if not hasattr(self, 'traj') or reset:
            self.reset(seed)
            print("Initializing and evaluating new population...")
            self.runtime = self.init_eval_pop(budget=budget)
            generation = generation + 1

        print("Running evolutionary search...")
        while not self.terminate:
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i + 1, generations, self.inc_score))
            runtime = self.evolve_generation(generation, budget=budget)
            self.runtime.extend(runtime)
            print(f"Generation:{generation}, Time: {sum(self.runtime)}")

            generation = generation + 1

        return np.array(self.runtime)
