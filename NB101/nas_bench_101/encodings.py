import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def encode_paths(path_indices):
    """ output one-hot encoding of paths """
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding

def encode_caz(matrix, ops):
    """ Path encoding expansion """
    encoding = {f"{op}-{in_out}-{i}":0 for in_out in ["in","out"] for op in ["conv3x3","conv1x1","maxpool3x3"] for i in range(1, 7)}
    encoding.update({f"in-out-{i}":0 for i in range(1, 7)})
    encoding.update({f"out-in-{i}":0 for i in range(1, 7)})

    for i in range(7):
        op = ops[i].split("-")[0]
        out_edges = matrix[i,:].sum()
        in_edges = matrix[:,i].sum()
        
        if ops[i] == INPUT and out_edges != 0:
            encoding[f"in-out-{out_edges}"] = 1
        elif ops[i] == OUTPUT and in_edges != 0:
            encoding[f"out-in-{in_edges}"] = 1
        else:
            if in_edges !=  0:
                encoding[f"{op}-in-{in_edges}"] = 1
            if out_edges != 0:
                encoding[f"{op}-out-{out_edges}"] = 1

    return np.array(list(encoding.values()))
