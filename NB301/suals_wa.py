import copy
import pickle
import numpy as np
from nas_benchmarks import Nasbench301
from nas_bench_301.cell_301 import Cell301

"""
    SuALS-WA
"""

class LocalSearch:

    def __init__(self, search_space, seed):

        # LS Parameters
        self.max_true = 3 # max no solns per iter to directly query true accuracy
        self.max_true_from_est = 5 # max no solns with estimated acc to  be selected for querying true acc per iter
        self.num_est = 10 # no solns with estimated accuracy per iter

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

        self.K = 1.6
        self.query = 0
        self.delta = 0.3
        self.seed = seed
        self.hashList = {}  # Store Generated solution hash values
        self.y_valid = list()
        self.network = dict()
        self.solutionList = dict()
        self.setAllTrueFitness = set()
        self.setAllEstimated = set()  # Stores the all estimated solution during one run
        self.search_space = search_space

    def reset(self):
        self.stepNo = 0
        self.query = 0
        self.inc_score = 0
        self.inc_config = None
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
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.neighbor_rnd = np.random.RandomState(self.seed)

    def save_data(self, data, filename):
        fh = open(f'results_output/{filename}.pkl', 'wb')
        pickle.dump(data, fh)
        fh.close()

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
    
    def best_solution_check(self, sol):
        if sol.fitness > self.inc_score:
            self.inc_score = sol.fitness
            self.inc_config = sol

    def getActualForSoln(self, soln):
        fitness = soln.get_val_loss(self.search_space.nasbench)
        self.y_valid.append(fitness)
        self.query += 1
        soln.fitness = fitness
        soln.fitnessType = "ACTUAL"
        self.setAllTrueFitness.add(soln)
        self.best_solution_check(soln)
        return fitness

    def init_random_samples(self, sample_size):
        samples = []
        for idx in range(sample_size):

            cell = Cell301(**self.get_cell())
            cell.solNo = self.solNo
            self.getActualForSoln(cell)

            cell.reliability = 1
            cell.upperLimit = 1
            self.solutionList[cell.solNo] = cell
            self.hashList[cell.get_hash()] = cell

            self.solNo += 1

            samples.append(cell)

        return samples
    
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
    
    
    def setSolTrueFitness(self, soln, setTrue):
        if soln.reliability is None:
            soln.reliability = 1
        soln.fitnessType = "ACTUAL"
        soln.upperLimit = 1
        setTrue.add(soln)
    
    
    def setSolEstimated(self, soln, setEstimated):
        if soln.reliability is None:
            soln.reliability = 0.5
        soln.fitnessType = "ESTIMATED"
        setEstimated.add(soln)

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
            if soln.fitness > self.inc_score:
                self.inc_score = soln.fitness
                self.inc_config = soln
        
        return self.inc_config # Return current incumbent

    def termination_check(self):
        if self.query > self.TOTAL_QUERIES:
            return True
        return False

    def mark_solution(self, neighbors, nonSameSolutions, type, setTrue, setEstimated):
        while len(neighbors) > 0:
            neighbor = neighbors.pop(0)['arch']
            cell = Cell301(arch=neighbor, arch_cont=None)
            if cell.get_hash() in self.hashList.keys(): # Same Solution Check
                continue
            cell.solNo = self.solNo
            self.solutionList[cell.solNo] = cell
            self.hashList[cell.get_hash()] = cell
            nonSameSolutions.append(cell)
            if type == 'ACTUAL':
                self.setSolTrueFitness(cell, setTrue)
                self.getActualForSoln(cell)
            else:
                self.setSolEstimated(cell, setEstimated)

            self.solNo += 1

            return self.termination_check(), cell
    
    def calculateReliabilityVals(self):
        # Converge edilecek çözümler
        isConverged = False
        reliabilityHistory = dict()
        convergenceList = copy.copy(self.setAllEstimated)
        while isConverged == False:

            reliabilityDiffList = []
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

    # Algorithm 5
    def calculateEstimateForSoln(self, network, solutionList):

        isConverged = False
        fitnessHistory = dict()
        convergenceList = copy.copy(self.setAllEstimated)

        for sol in self.setAllEstimated:
            if sol.fitnessType == "ESTIMATED" and sol.fitness == 0:
                sol.fitness = 0.5

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

        for soln in solutions:
            self.network.setdefault(soln.solNo, dict())
            if soln.fitnessType == 'ESTIMATED' and len(self.network[soln.solNo]) < 1: # Isolated Estimated Solutions
                del self.network[soln.solNo]
                del self.solutionList[soln.solNo]
                self.setAllEstimated -= {soln}
                setEstimated -= {soln}
        
        self.setAllEstimated |= setEstimated  # Merge the two sets
        self.nbrEstimatedSoln += len(setEstimated)

        # Reliability Calculation
        self.calculateReliabilityVals()
        
        # // Steps 7 and 8 in Figure 1
        self.calculateEstimateForSoln(self.network, self.solutionList)

    def estimated_to_true(self, sol, setEstimated, setTrue):
        self.getActualForSoln(sol)
        self.setSolTrueFitness(sol, setTrue)
        self.nbrActualSoln += 1
        sol.reliability = 1
        setEstimated -= {sol}
        self.setAllEstimated -= {sol}

        # Termination Condition
        if self.query > self.TOTAL_QUERIES:
            return True

        return False

    def next_step(self, incumbent):
        neighbors = incumbent.get_neighborhood(self.search_space.nasbench, rnd_generator=self.neighbor_rnd)
        return set(), set(), [], neighbors

    def local_search(self):
        
        print("Initializing Phase...")
        incumbent = self.init_phase(sample_size=10)
        neighbors = incumbent.get_neighborhood(self.search_space.nasbench, rnd_generator=self.neighbor_rnd)

        self.i = 0
        ctr_true = 0
        ctr_true_from_est = 0
        nonSameSolutions = []
        setTrue = set()
        setEstimated = set()
        print("Local Search...")
        while self.query < self.TOTAL_QUERIES:
            
            if ctr_true < self.max_true:
                terminate, v_i = self.mark_solution(neighbors, nonSameSolutions, 'ACTUAL', setTrue, setEstimated)
                if terminate:
                    break
                self.true_step(v_i)
                ctr_true += 1
            else:
                if ctr_true_from_est == 0:
                    solutions = []
                    while len(solutions) < self.num_est and len(neighbors) > 0:
                        terminate, cell = self.mark_solution(neighbors, nonSameSolutions, 'ESTIMATED', setTrue, setEstimated)
                        solutions.append(cell)
                    self.estimated_step(setEstimated, solutions)

                ctr_true_from_est += 1
                v_i = max(setEstimated, key=lambda x: x.fitness) # best estimated
                self.estimated_to_true(v_i, setEstimated, setTrue)
                # Termination Condition 
                if self.termination_check():
                    break
            
            if v_i.fitness > incumbent.fitness:
                self.i += 1
                ctr_true = 0
                ctr_true_from_est = 0
                incumbent = v_i
                setTrue, setEstimated, nonSameSolutions, neighbors = self.next_step(incumbent)
                
            elif ctr_true_from_est == self.max_true_from_est:
                incumbent = max(self.setAllTrueFitness - {incumbent}, key=lambda x: x.fitness)
                self.i += 1
                ctr_true = 0
                ctr_true_from_est = 0
                setTrue, setEstimated, nonSameSolutions, neighbors = self.next_step(incumbent)
                
            if self.query % 10 == 0:
                print(self.query, self.inc_score)

    def run(self):
        self.solNo = 0

        self.reset()

        print("Seed:", self.seed)
        self.local_search()
        

search_space = Nasbench301()

for i in range(500):
    ls = LocalSearch(search_space, seed=i)
    ls.run()
