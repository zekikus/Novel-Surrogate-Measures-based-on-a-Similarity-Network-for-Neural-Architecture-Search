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
        SuADE-RG
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
        self.n_min = 40
        self.T = 3
        self.delta = 0.5
        self.threshold = 0.1
        self.db = kwargs['b']
        self.hashList = {}  # Store Generated solution hash values
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
        self.hashList = {}

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

        print(self.mutation_strategy)
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
            if cell.solNo == solNo: continue

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
        totalTime = sum(self.runtime)  # Tüm jenerasyonlar için harcanan true fitness hesaplama süresi
        generationCost = 0  # İlgili jenerasyonda true fitness hesaplaması için geçen süre

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
                self.setSolTrueFitness(soln, setTrue, setEstimated)

        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrActualSoln += len(setTrue)
        self.nbrEstimatedSoln += len(setEstimated)

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

        for soln in setEstimated:
            self.network.setdefault(soln.solNo, dict())
            if soln.fitnessType == 'ESTIMATED' and len(self.network[soln.solNo]) < 1: # Isolated Estimated Solutions
                del self.network[soln.solNo]
                del self.solutionList[soln.solNo]
                setEstimated -= {soln}

        true_neighbors = set() # set R
        for soln in setEstimated:
            true_neighbors |= {n for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].fitness != 0}

        # Create/Update Regression Model
        for soln in true_neighbors:
            self.regression_model(self.solutionList[soln])

        for soln in setEstimated:
            nbr_true_neighbors = self.calculateEstimateForSoln(soln)
           
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
    def calculateEstimateForSoln(self, soln):

        regression_models = [self.solutionList[n] for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].regression_model is not None and self.solutionList[n].fitness != 0]

        if len(regression_models) > 0:
            soln.fitness = (sum(n.regression_model.predict([[self.network[soln.solNo][n.solNo]]])[0][0] for n in regression_models if n.regression_model is not None) / len(regression_models))
        else:
            soln.fitness = self.getEstimatedFitness(soln)

        return len(regression_models)

    # Algorithm 3
    def updateNetwork(self, edge, solutionList, nonSameSolutions, network, setTrue, setEstimated):
        v = None
        w = None
        node1, node2, distance = solutionList[edge[0]], solutionList[edge[1]], edge[2]

        self.network.setdefault(node1.solNo, dict())
        self.network.setdefault(node2.solNo, dict())     
            
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
                self.setSolEstimated(v, setEstimated, setTrue)
            # If w ∈ N, one of v or w is randomly assigned to T and the other is added to E
            elif w in nonSameSolutions:
                solutions = [w, v]
                self.type_decision_rnd.shuffle(solutions)
                # Add True Fitness List
                setEstimated |= set([solutionList[n] for n in network[solutions[0].solNo].keys() if solutionList[n].fitnessType == "ESTIMATED"])
                self.setSolTrueFitness(solutions[0], setTrue, setEstimated)
                # Add Estimated List
                self.setSolEstimated(solutions[1], setEstimated, setTrue)
            # If w ∈ E, then v is added to T if ˜A(w) > α, and added to E, otherwise.
            elif w.fitnessType == "ESTIMATED":
                fitness = self.calculateEstimateForSoln(w)
                if fitness > max(self.inc_score - self.threshold, 0.6):  # Threshold Check
                    self.setSolTrueFitness(v, setTrue, setEstimated)
                    v.reliability = 1
                    self.regression_model(v)
                    # Add neighbors of v to the setEstimated
                    neighbors = [solutionList[n] for n in network[v.solNo].keys() if solutionList[n].fitnessType == "ESTIMATED"]
                    setEstimated |= set(neighbors)
                else:
                    self.setSolEstimated(v, setEstimated, setTrue)

        return self.network, setEstimated, setTrue

    # Algorithm 4
    def getActualForSoln(self, soln, solutionList):
        fitness, cost = self.f_objective(soln, budget=108, addResultFile=True)
        soln.fitness = fitness
        soln.fitnessType = "ACTUAL"
        soln.cost = cost
        soln.upperLimit = 1

        return cost, fitness
    
    def regression_model(self, soln):
        true_neighbors = [self.solutionList[n] for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].fitness != 0]
        if len(true_neighbors) >= self.n_min:
            X = [[self.network[soln.solNo][n.solNo]] for n in true_neighbors]
            y = [[n.fitness] for n in true_neighbors]
            soln.fit_regression_model(X, y)

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
            self.setSolTrueFitness(soln, setTrue, setEstimated)

        for soln, d in candidateTrueFitness[self.T:]:
            del self.network[soln.solNo]
            del self.solutionList[soln.solNo]
            soln.fitness = 0

        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrActualSoln += len(setTrue)
        self.nbrEstimatedSoln += len(setEstimated)

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

        for soln in setEstimated:
            self.network.setdefault(soln.solNo, dict())
            if soln.fitnessType == 'ESTIMATED' and len(self.network[soln.solNo]) < 1: # Isolated Estimated Solutions
                del self.network[soln.solNo]
                del self.solutionList[soln.solNo]
                setEstimated -= {soln}

        true_neighbors = set() # set R
        for soln in setEstimated:
            true_neighbors |= {n for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].fitness != 0}

        # Create/Update Regression Model
        for soln in true_neighbors:
            self.regression_model(self.solutionList[soln])

        for soln in setEstimated:
            nbr_true_neighbors = self.calculateEstimateForSoln(soln)
            
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
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(generation + 1, generations, self.inc_score))
            runtime = self.evolve_generation(generation, budget=budget)
            self.runtime.extend(runtime)
            print(f"Generation:{generation}, Time: {sum(self.runtime)}")

            generation = generation + 1

        if verbose:
            print("\nRun complete!")

        return np.array(self.runtime)
