'''Runs SADE on NAS-Bench-101
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import json
import pickle
import argparse
import numpy as np

from suade.suade_wa import SADE
from tabular_benchmarks import NASCifar10A

def save_history(history, path, filename='history'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(history, fh)
    fh.close()

def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()

# Common objective function for SADE representing NAS-Bench-101
def f(cell, budget=None, addResultFile=None):
    if budget is not None:
        fitness, cost = b.objective_function(cell, budget=int(budget), addResultFile=addResultFile)
    else:
        fitness, cost = b.objective_function(cell)
    return fitness, cost


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=500, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
choices = ["nas_cifar10a", "nas_cifar10b", "nas_cifar10c"]
parser.add_argument('--benchmark', default="nas_cifar10a", type=str,
                    help="specify the benchmark to run on from among {}".format(choices))
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas_benchmarks-development/"
                                          "tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=30, type=int, nargs='?', help='population size')
strategy_choices = ['rand1_bin', 'rand2_bin', 'rand2dir_bin', 'best1_bin', 'best2_bin',
                    'currenttobest1_bin', 'randtobest1_bin',
                    'rand1_exp', 'rand2_exp', 'rand2dir_exp', 'best1_exp', 'best2_exp',
                    'currenttobest1_exp', 'randtobest1_exp']
parser.add_argument('--strategy', default="currenttobest1_bin", choices=strategy_choices,
                    type=str, nargs='?',
                    help="specify the DE strategy from among {}".format(strategy_choices))
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                    help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                    help='probability of crossover')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='result', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.benchmark == "nas_cifar10a": # NAS-Bench-101
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

# Parameter space to be used by SADE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.folder)
os.makedirs(output_path, exist_ok=True)

# Initializing SADE object
sade = SADE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
        mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
        strategy=args.strategy, budget=max_budget, b=b, output_path=output_path)

for run_id, _ in enumerate(range(args.runs), start=args.run_start):
    if not args.fix_seed:
        print("SEED:", run_id)
        np.random.seed(run_id)
    if args.verbose:
        print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))

    # Running SADE iterations
    if run_id == 0:
        runtime = sade.run(generations=args.gens, verbose=args.verbose, seed=run_id)
    else:
        runtime = sade.run(generations=args.gens, verbose=args.verbose, seed=run_id)

    if 'cifar' in args.benchmark:
        res = b.get_results(ignore_invalid_configs=True)
    else:
        res = b.get_results()
    
    # Save SADE Run
    fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
    json.dump(res, fh)
    fh.close()
    if args.verbose:
        print("Run saved. Resetting...")
    # essential step to not accumulate consecutive runs
    b.reset_tracker()