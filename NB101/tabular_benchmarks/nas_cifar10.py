import os
import copy
import ConfigSpace
import numpy as np
from nasbench import api
from nasbench.lib import graph_util

MAX_EDGES = 9
VERTICES = 7


class NASCifar10(object):

    def __init__(self, data_dir, multi_fidelity=True):

        self.multi_fidelity = False

        self.dataset = api.NASBench(os.path.join('/home/zekikus/Desktop/NASNet/NAS-101/nasbench/', 'nasbench_only108.tfrecord'))
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []

        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []

    @staticmethod
    def objective_function(self, config):
        pass

    def record_invalid(self, config, valid, test, costs):
        self.X.append(config)
        self.y_valid.append(valid)
        self.y_test.append(test)
        self.costs.append(costs)

    def record_valid(self, config, data, model_spec):

        self.X.append(config)

        # compute mean test error for the final budget
        _, metrics = self.dataset.get_metrics_from_spec(model_spec)
        mean_test_error = 1 - np.mean([metrics[108][i]["final_test_accuracy"] for i in range(3)])
        self.y_test.append(mean_test_error)

        # compute validation error for the chosen budget
        valid_error = 1 - data["validation_accuracy"]
        self.y_valid.append(valid_error)

        runtime = data["training_time"]
        self.costs.append(runtime)

    @staticmethod
    def get_configuration_space():
        pass

    def get_results(self, ignore_invalid_configs=False):

        regret_validation = []
        regret_test = []
        runtime = []
        rt = 0

        inc_valid = np.inf
        inc_test = np.inf

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]
                inc_test = self.y_test[i]

            regret_validation.append(float(inc_valid - self.y_star_valid))
            regret_test.append(float(inc_test - self.y_star_test))
            rt += self.costs[i]
            runtime.append(float(rt))

        res = dict()
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['runtime'] = runtime

        return res


class NASCifar10A(NASCifar10):
    def objective_function(self, cell, budget=108, addResultFile=None):
        if self.multi_fidelity is False:
            assert budget == 108

        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(cell.matrix) > MAX_EDGES:
            self.record_invalid(cell.config, 0, 0, 0)
            #print("Invalid Architecture...")
            return 0, 0

        labeling = copy.copy(cell.ops)
        model_spec = api.ModelSpec(cell.matrix, labeling)
        try:
           data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(cell.config, 0, 0, 0)
            #print("Invalid Architecture...")
            return 0, 0

        if addResultFile == True:
            self.record_valid(cell.config, data, model_spec)
        return data["validation_accuracy"], data["training_time"]
    
    def feasibilityCheck(self, cell):
        if graph_util.num_edges(cell.matrix) > MAX_EDGES:
            return False
        model_spec = api.ModelSpec(cell.matrix, cell.ops)
        return self.dataset.is_valid(model_spec)
    
    def getUsedNodes(self, cell):
        matrix = cell.matrix
        config = cell.config
        
        used_nodes = []
        for i in range(VERTICES - 2):
            _sum = np.sum(matrix[:, i + 1])
            if _sum > 0:
                used_nodes.append((i + 1, config[f"op_node_{i}"]))

        return used_nodes

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs


class NASCifar10B(NASCifar10):
    def objective_function(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        bitlist = [0] * (VERTICES * (VERTICES - 1) // 2)
        for i in range(MAX_EDGES):
            bitlist[config["edge_%d" % i]] = 1
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit

        matrix = np.fromfunction(graph_util.gen_is_edge_fn(out),
                                 (VERTICES, VERTICES),
                                 dtype=np.int8)
        # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        self.record_valid(config, data, model_spec)

        return 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        cat = [i for i in range((VERTICES * (VERTICES - 1)) // 2)]
        for i in range(MAX_EDGES):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, cat))
        return cs


class NASCifar10C(NASCifar10):
    def objective_function(self, config, budget=108):
        if self.multi_fidelity is False:
            assert budget == 108

        edge_prob = []
        for i in range(VERTICES * (VERTICES - 1) // 2):
            edge_prob.append(config["edge_%d" % i])

        idx = np.argsort(edge_prob)[::-1][:config["num_edges"]]
        binay_encoding = np.zeros(len(edge_prob))
        binay_encoding[idx] = 1
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = binay_encoding[i]

        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)
        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        self.record_valid(config, data, model_spec)

        return 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))

        cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("num_edges", 0, MAX_EDGES))

        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("edge_%d" % i, 0, 1))
        return cs
