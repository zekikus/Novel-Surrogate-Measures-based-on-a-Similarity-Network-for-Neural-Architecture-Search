import copy
import pickle
import numpy as np
from nas_benchmarks import Nasbench301 # TEST için yazıldı silinecek
from nas_bench_301.cell_301 import Cell301

class DE:
    """
        SuADE-RG
    """
    def __init__(self, mutation_factor=0.5, crossover_prob=0.5, cutoff=8e4, 
                       pop_size=30, seed=None, search_space=None):
        
        self.network = dict()
        self.solutionList = dict()
        self.setAllEstimated = set()  # Stores the all estimated solution during one run
        self.setAllTrueFitness = set()
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.cutoff = cutoff
        self.pop_size = pop_size
        self.population = []
        self.seed = seed
        self.query = 0
        self.hashList = {}  # Store Generated solution hash values
        self.search_space = search_space
        self.n_min = 30
        self.T = 3 # A*'a en yakın T çözümü isolated kümesinden trueFitness kümesine al
        self.delta = 0.4
        self.num_init = 10
        self.threshold = 0.0025

        # CONSTANTS
        self.NUM_OPS = 7
        self.NUM_VERTICES = 4
        self.TOTAL_QUERIES = 150
        self.dimensions = 16

        # Global trackers
        self.inc_score = 0
        self.inc_config = None
        self.nbrEstimatedSoln = 0
        self.nbrActualSoln = 0
        self.y_valid = list() # True Fitness değeri hesaplanan tüm çözümlerin accuracy değerlerini saklar

        self.reset()

    def save_data(self, data, filename):
        fh = open(f'results_output/{filename}.pkl', 'wb')
        pickle.dump(data, fh)
        fh.close()

    def reset(self):
        self.query = 0
        self.inc_score = 0
        self.inc_config = None
        self.population = None
        self.y_valid = list()
        self.network = dict()
        self.solutionList = dict()
        self.setAllEstimated = set()
        self.hashList = {}
        self.nbrEstimatedSoln = 0
        self.nbrActualSoln = 0
        self.initRndNumberGenerators()

    def initRndNumberGenerators(self):
        # Random Number Generators
        self.disc_cont_rnd = np.random.RandomState(self.seed)
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.type_decision_rnd = np.random.RandomState(self.seed)
        self.neighbor_rnd = np.random.RandomState(self.seed)

    # Generate Random Cell
    def get_cell(self, normal_cont = None, reduction_cont = None):
        # [(), (), (), (), (), (), (), ()]
        normal = []
        reduction = []

        if normal_cont is None and reduction_cont is None:
            normal_cont = self.sample_pop_rnd.uniform(low=0.0, high=1.0, size=self.dimensions)
            reduction_cont = self.sample_pop_rnd.uniform(low=0.0, high=1.0, size=self.dimensions)
        
        for i in range(self.NUM_VERTICES):
            normal.extend(self.continuous_to_discrete(i, [normal_cont[idx] for idx in range(i*4, (i+1) * 4)]))
            reduction.extend(self.continuous_to_discrete(i, [reduction_cont[idx] for idx in range(i*4, (i+1) * 4)]))

        return {'arch': (normal, reduction), 'arch_cont': (normal_cont, reduction_cont)}

    def initialization_check(self, cell, population):
        isSameSolution, solNo = self.checkSolution(cell)
        if not isSameSolution:
            self.solNo = self.solNo + 1
            cell.solNo = self.solNo
            cell.reliability = 1
            self.solutionList[cell.solNo] = cell
            self.hashList[cell.get_hash()] = cell
            self.getActualForSoln(cell)
            self.network.setdefault(cell.solNo, dict())
            normal_cell = tuple([tuple(l) for l in cell.arch[0]])
            reduction_cell = tuple([tuple(l) for l in cell.arch[1]])
            cell.arch = (normal_cell, reduction_cell)
            population.append(cell)
        
        return isSameSolution


    def init_random_samples(self):
        population = list()
        print("Init Population: ", self.pop_size)
        while len(population) < self.num_init:
            cell = Cell301(**self.get_cell())
            self.initialization_check(cell, population)
        
        bestArch = max(population, key=lambda x: x.fitness)
        while len(population) < self.pop_size:
            nbhd = bestArch.get_neighborhood(self.search_space.nasbench, rnd_generator=self.neighbor_rnd)
            for nbr in nbhd:
                cell = Cell301(**self.get_rev_cell(nbr["arch"][0], nbr["arch"][1]))
                isSameSolution = self.initialization_check(cell, population)
                if (not isSameSolution) and cell.fitness > bestArch.fitness:
                    bestArch = cell
                    break
                
                if len(population) > self.pop_size:
                    break

        return np.array(population)

    # Local Search'den gelen komşulukları continous vektörlere çevirmek için yazıldı
    def get_rev_cell(self, normal_disc, reduction_disc):
        normal = []
        reduction = []
        for i in range(self.NUM_VERTICES):
            normal.extend(self.discrete_to_continous(normal_disc, i))
            reduction.extend(self.discrete_to_continous(reduction_disc, i))

        return {'arch': (normal_disc, reduction_disc), 'arch_cont': (np.array(normal), np.array(reduction))}

    # Convert continuous vector to discrete values 
    def discrete_to_continous(self, discrete_values, node_idx):
        ops_bins = np.linspace(0, 1, num = self.NUM_OPS + 1)

        in_edges = discrete_values[node_idx * 2: node_idx * 2 + 2]
        nodes = [i for i in range(node_idx + 2)]
        result = list()
            
        for i in range(2):
            bins = np.linspace(0, 1, num = len(nodes) + 1)
            low = bins[nodes.index(in_edges[i][0])]
            if len(bins) < 3: # 3'den az aralık olursa hata veriyordu
                low = 0
                high = 1
            else:
                high = bins[min(nodes.index(in_edges[i][0]) + 1, len(bins) - 1)]
            node_cont_val = self.disc_cont_rnd.uniform(low, high)
            
            low = ops_bins[in_edges[i][1]]
            high = ops_bins[min(in_edges[i][1] + 1, len(ops_bins))]
            ops_cont_val = self.disc_cont_rnd.uniform(low, high)
            
            result.append(node_cont_val)
            result.append(ops_cont_val)
            nodes.pop(nodes.index(in_edges[i][0]))

        return result

    # Convert continuous vector to discrete values 
    def continuous_to_discrete(self, idx, cont_values):
        nodes = [i for i in range(idx + 2)]
        ops = [i for i in range(self.NUM_OPS)]
        ops_bins = np.linspace(0, 1, num = self.NUM_OPS + 1)
        result = list()
        for i in range(2):
            bins = np.linspace(0, 1, num = len(nodes) + 1)
            selectedNode = np.digitize(cont_values[i * 2], bins, right=True) - 1 # Input Node
            selectedOp = np.digitize(cont_values[i * 2 + 1], ops_bins, right=True) - 1 # Operation Edge
            result.append((nodes[selectedNode], ops[selectedOp]))
            nodes.pop(selectedNode)

        return result
    
    def init_eval_pop(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_random_samples()
        # Best Solution
        bestArch = max(self.population, key=lambda x: x.fitness)
        self.inc_score = bestArch.fitness
        self.inc_config = bestArch
        ########

        setTrue = copy.deepcopy(self.population)
        # Create Candidate Edge List
        candidateEdges = set()  # the set of candidate edges
        for soln in self.population:
            candidateEdges = self.getSolutionCandidateEdges(soln, self.solutionList, self.delta, candidateEdges)
        
        # The edges in C are sorted in increasing distance
        candidateEdges = sorted(candidateEdges, key=lambda x: x[2])

        # Update Network
        for edge in candidateEdges:
            node1, node2, distance = self.solutionList[edge[0]], self.solutionList[edge[1]], edge[2]

            self.network.setdefault(node1.solNo, dict())
            self.network.setdefault(node2.solNo, dict())

            self.network[node1.solNo][node2.solNo] = distance
            self.network[node2.solNo][node1.solNo] = distance
        
        self.nbrActualSoln += len(setTrue)

        for soln in self.population:
            # Best Solution
            if soln.fitness > self.inc_score:
                self.inc_score = soln.fitness
                self.inc_config = soln
        
        return

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
        """
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.
        """
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        vector[violations] = [1 - (vector[i] - 1) if vector[i] > 1 else abs(vector[i]) for i in violations]
        return vector

    def mutation(self, current, best):
        '''Performs DE mutation
        '''
        r1, r2 = self.sample_population(size=2, alt_pop=None)
        mutant = self.mutation_currenttobest1(current, best, r1, r2)

        return np.array(mutant)
    
    def mutation_currenttobest1(self, current, best, r1, r2):
        '''Performs the 'current_to_best1' type of DE mutation
        '''
        mutant_cont = list() # [normal_cont, reduction_cont]
        for idx in range(2):
            diff1 = best.arch_cont[idx] - current.arch_cont[idx]
            diff2 = r1.arch_cont[idx] - r2.arch_cont[idx]
            mutant = current.arch_cont[idx] + self.mutation_factor * diff1 + self.mutation_factor * diff2
            mutant = self.boundary_check(mutant)
            mutant_cont.append(mutant)
        
        return mutant_cont
    
    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        offspring = [None, None]
        offspring[0] = self.crossover_bin(target.arch_cont[0], mutant[0])
        offspring[1] = self.crossover_bin(target.arch_cont[1], mutant[1])

        return offspring
    
    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def checkSolution(self, cell):
        for solNo, solRef in self.solutionList.items():
            D = cell.distance(solRef, dist_type='caz')
            if D == 0:
                return True, solNo

        return False, cell.solNo
    
    # Get candidate edges for given solution as parameter
    def getSolutionCandidateEdges(self, soln, solutionList, delta, candidateEdges):
        for indvNo, indv in solutionList.items():
            if indv.solNo == soln.solNo:
                continue

            # Calculate Jackard Distance between indv and soln
            distance = soln.distance(indv, dist_type='caz')
            if distance < delta:
                minSolNo = min(indv.solNo, soln.solNo)
                maxSolNo = max(indv.solNo, soln.solNo)
                candidateEdges.add((minSolNo, maxSolNo, distance))

        return candidateEdges
    
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
                if fitness > max(self.inc_score - self.threshold, 0.94):  # Threshold Check
                    self.setSolTrueFitness(v, setTrue, setEstimated)
                    v.reliability = 1
                    self.regression_model(v)
                    # Add neighbors of v to the setEstimated
                    neighbors = [solutionList[n] for n in network[v.solNo].keys() if solutionList[n].fitnessType == "ESTIMATED"]
                    setEstimated |= set(neighbors)
                else:
                    self.setSolEstimated(v, setEstimated, setTrue)

        return self.network, setEstimated, setTrue
    
    # Çözümü true fitness olarak işaretle
    def setSolTrueFitness(self, soln, setTrue, setEstimated):
        if soln.reliability is None:
            soln.reliability = 1
        soln.fitnessType = "ACTUAL"
        soln.upperLimit = 1
        setTrue.add(soln)
        setEstimated -= {soln}
    
    # Çözümü estimated olarak işaretle
    def setSolEstimated(self, soln, setEstimated, setTrue):
        if soln.reliability is None:
            soln.reliability = 0.5
        soln.fitnessType = "ESTIMATED"
        setEstimated.add(soln)
        setTrue -= {soln}
    
    # Algorithm 4
    def getActualForSoln(self, soln):
        fitness = soln.get_val_loss(self.search_space.nasbench)
        self.y_valid.append(fitness)
        self.query += 1
        soln.fitness = fitness
        soln.fitnessType = "ACTUAL"
        soln.upperLimit = 1
        self.setAllTrueFitness.add(soln)

        return fitness

    def getEstimatedFitness(self, soln):
        numerator = 0
        denominator = 0
        for solnNo, distance in self.network[soln.solNo].items():
            if self.solutionList[solnNo].fitness != 0:
                numerator += ((1 - distance) * self.solutionList[solnNo].fitness)
                denominator += (1 - distance)
        
        return numerator / denominator

    # Algorithm 5
    def calculateEstimateForSoln(self, soln):

        regression_models = [self.solutionList[n] for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].regression_model is not None and self.solutionList[n].fitness != 0]

        if len(regression_models) > 0:
            soln.fitness = (sum(n.regression_model.predict([[self.network[soln.solNo][n.solNo]]])[0][0] for n in regression_models if n.regression_model is not None) / len(regression_models))
            soln.estimation_type = 'regression'
        else:
            soln.fitness = self.getEstimatedFitness(soln)
            soln.estimation_type = 'weighted_avg'

        return len(regression_models)

    def regression_model(self, soln):
        true_neighbors = [self.solutionList[n] for n in self.network[soln.solNo].keys() if self.solutionList[n].fitnessType == 'ACTUAL' and self.solutionList[n].fitness != 0]
        if len(true_neighbors) >= self.n_min:
            X = [[self.network[soln.solNo][n.solNo]] for n in true_neighbors]
            y = [[n.fitness] for n in true_neighbors]
            soln.fit_regression_model(X, y)

    def evolve_generation(self, generation):
        '''
        Performs a complete SADE evolution
        '''

        trials = []  # İlgili jenerasyonda üretilen tüm çözümleri saklar (Aynı olan çözümler dahil)
        nonSameSolutions = []  # İlgili jenerasyonda üretilen ve aynı olmayan çözümleri saklar
        Pnext = []
        runtime = []
        setTrue = set()
        setEstimated = set()

        generationBest = max(self.population, key=lambda x: x.fitness)

        for j in range(self.pop_size):
            target = self.population[j]
            mutant = copy.deepcopy(target)
            mutant = self.mutation(target, generationBest)
            mutant = self.crossover(target, mutant)
            cell = Cell301(**self.get_cell(normal_cont=mutant[0], reduction_cont=mutant[1]))
            self.solNo += 1
            cell.solNo = self.solNo

            # Isomorphism Check
            isSameSolution, solNo = self.checkSolution(cell)
            if isSameSolution:
                cell = self.solutionList[solNo]
            else:
                self.solutionList[cell.solNo] = cell
                self.hashList[cell.get_hash()] = cell
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
                distance = soln.distance(self.inc_config, dist_type='caz')
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
            fitness = self.getActualForSoln(soln)

            # Best Solution
            if fitness > self.inc_score:
                self.inc_score = fitness
                self.inc_config = soln

            # Termination Condition
            if self.query > self.TOTAL_QUERIES:
                self.terminate = True
                return

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
        
        # v1 - Ek
        if len(setEstimated) > 0:
            bestEstimated = max(setEstimated, key=lambda x: x.fitness)
            
            # Remove bestEstimated solution from AllEstimated set
            self.setAllEstimated = self.setAllEstimated - {bestEstimated}
            self.setSolTrueFitness(bestEstimated, setTrue, setEstimated)
            self.getActualForSoln(bestEstimated)
            bestEstimated.reliability = 1
        ########

        return

    def run(self, reset=True, seed=None):
        generation = 1  # Generation Number
        self.seed = seed
        self.solNo = 0
        self.terminate = False

        # checking if a run exists
        if not hasattr(self, 'traj') or reset:
            self.reset()
            print("Initializing and evaluating new population...")
            self.init_eval_pop()
            generation = generation + 1

        print("Running evolutionary search...")
        while not self.terminate:
            self.evolve_generation(generation)
            if generation > 10:
                print(f"Generation:{generation}, Query: {self.query}")

            generation = generation + 1

        return

search_space = Nasbench301()
for i in range(500):
    print(f"Run {i}")

    de = DE(seed=i, search_space=search_space, pop_size=30)
    de.run(seed=i)