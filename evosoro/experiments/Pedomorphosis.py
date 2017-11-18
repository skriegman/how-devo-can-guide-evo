import re
import random
import sys
import os
import numpy as np
import subprocess as sub
from glob import glob
import ast

sys.path.append(os.getcwd() + "/../..")

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import DirectEncoding
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import SetMutRateOptimization


sub.call("cp ~/pkg/evosoro/evosoro/_voxcad/voxelyzeMain/voxelyze .", shell=True)
# sub.call("cp ../_voxcad/voxelyzeMain/voxelyze .", shell=True)


SEED = int(sys.argv[1])
MAX_TIME = float(sys.argv[2])

POP_SIZE = None  # defined below based on length of lineage
IND_SIZE = (4, 4, 3)
MAX_GENS = 0  # just evaluate the initial pop (the lineage of the run champion)
NUM_RANDOM_INDS = 0

SIM_TIME = 10.5  # includes init time
INIT_TIME = 0.5

DT_FRAC = 0.35
MIN_TEMP_FACT = 0.25
GROWTH_AMPLITUDE = 0.75
MIN_GROWTH_TIME = 0.10
TEMP_AMP = 39

MUT_NET_PROB_DIST = [1, 0.25, 0.25, 0.25, 0.25]  # first prob is for meta mutation
MUT_RATE = 1/48.  # starting and lower bound of meta mutation rate (one voxel)
MUT_SCALE = 1
META_MUT_RATE = 48/48.
META_MUT_SCALE = 4/48.

STOP_AFTER_FALLING = True

NET_DICT = {"init_size": [], "final_size": [], "init_offset": [], "final_offset": []}

TIME_TO_TRY_AGAIN = 12
MAX_EVAL_TIME = 30

SAVE_VXA_EVERY = MAX_GENS+1  # never save
SAVE_LINEAGES = False
CHECKPOINT_EVERY = 100
EXTRA_GENS = 0

RUN_DIR = "run_{}".format(SEED)
RUN_NAME = "RollTime"

run_directory = "/users/s/k/skriegma/scratch/AFPO_Both_M_48_4/run_{}/".format(SEED)
# run_directory = "/home/sam/Archive/skriegma/AFPO_Both_M_48_4/run_{}/".format(SEED)
all_gen_files = glob(run_directory + "allIndividualsData/Gen_*.txt")
sorted_gen_files = sorted(all_gen_files, reverse=True)

first_ancestor = False
run_champ_id, run_champ_fitness = -1, -1
lineage = []
gens_back = 0
while not first_ancestor:

    with open(sorted_gen_files[gens_back], 'r') as infile:
        for line in infile:
            if line.split()[0] != 'gen':
                this_id = int(line.split()[1])
                parent_id = int(line.split()[3])
                if len(lineage) == 0 or this_id == lineage[-1]:
                    lineage += [parent_id]

                    s = ast.literal_eval(re.split(r'\t+', line)[33])
                    f = ast.literal_eval(re.split(r'\t+', line)[34])
                    s, f = np.array(s), np.array(f)
                    w = np.sum(np.abs(f-s))
                    true = float(line.split()[28])
                    assert(abs(true-w) < 10e-9)

                    s_cube = np.resize(s, IND_SIZE)
                    reversed_array = s_cube[::-1, :, :]
                    if np.all(s_cube[:int(IND_SIZE[0]/2.0), :, :] == reversed_array[:int(IND_SIZE[0]/2.0), :, :]):
                        NET_DICT['final_size'] += [np.reshape(f, IND_SIZE)]
                        NET_DICT['init_size'] += [np.reshape(s, IND_SIZE)]

                        s1 = ast.literal_eval(re.split(r'\t+', line)[31])
                        f1 = ast.literal_eval(re.split(r'\t+', line)[32])
                        s1, f1 = np.array(s1), np.array(f1)
                        w1 = np.sum(np.abs(f1 - s1))
                        true = float(line.split()[25])
                        assert (abs(true - w1) < 10e-9)

                        NET_DICT['final_offset'] += [np.reshape(f1, IND_SIZE)]
                        NET_DICT['init_offset'] += [np.reshape(s1, IND_SIZE)]

                    else:
                        NET_DICT['final_offset'] += [np.reshape(f, IND_SIZE)]
                        NET_DICT['init_offset'] += [np.reshape(s, IND_SIZE)]

                        s1 = ast.literal_eval(re.split(r'\t+', line)[31])
                        f1 = ast.literal_eval(re.split(r'\t+', line)[32])
                        s1, f1 = np.array(s1), np.array(f1)
                        w1 = np.sum(np.abs(f1 - s1))
                        true = float(line.split()[25])
                        assert (abs(true - w1) < 10e-9)

                        NET_DICT['final_size'] += [np.reshape(f1, IND_SIZE)]
                        NET_DICT['init_size'] += [np.reshape(s1, IND_SIZE)]

                        s1_cube = np.resize(s1, IND_SIZE)
                        reversed_array1 = s1_cube[::-1, :, :]
                        if not np.all(s1_cube[:int(IND_SIZE[0]/2.0), :, :] == reversed_array1[:int(IND_SIZE[0]/2.0), :, :]):
                            raise AssertionError

                    if parent_id == -1:
                        first_ancestor = True

    gens_back += 1


POP_SIZE = len(NET_DICT['init_size'])


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(DirectEncoding(output_node_name="mutation_rate", orig_size_xyz=IND_SIZE,
                                        scale=META_MUT_SCALE, p=META_MUT_RATE, symmetric=False,
                                        lower_bound=MUT_RATE, start_val=MUT_RATE, mutate_start_val=True))

        self.add_network(DirectEncoding(output_node_name="init_size", orig_size_xyz=IND_SIZE,
                                        scale=MUT_SCALE))
        self.to_phenotype_mapping.add_map(name="init_size", tag="<InitialVoxelSize>")

        self.add_network(DirectEncoding(output_node_name="final_size", orig_size_xyz=IND_SIZE,
                                        scale=MUT_SCALE))
        self.to_phenotype_mapping.add_map(name="final_size", tag="<FinalVoxelSize>")

        self.add_network(DirectEncoding(output_node_name="init_offset", orig_size_xyz=IND_SIZE,
                                        scale=MUT_SCALE, symmetric=False))
        self.to_phenotype_mapping.add_map(name="init_offset", tag="<PhaseOffset>")

        self.add_network(DirectEncoding(output_node_name="final_offset", orig_size_xyz=IND_SIZE,
                                        scale=MUT_SCALE, symmetric=False))
        self.to_phenotype_mapping.add_map(name="final_offset", tag="<FinalPhaseOffset>")


random.seed(SEED)
np.random.seed(SEED)

my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT,
             fitness_eval_init_time=INIT_TIME)

my_env = Env(temp_amp=TEMP_AMP, falling_prohibited=STOP_AFTER_FALLING)
my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")

my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<finalDistY>")
my_objective_dict.add_objective(name="lifetime", maximize=True, tag="<Lifetime>", logging_only=True)
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POP_SIZE)

names = [k for k, v in NET_DICT.items()]
nets = [v for k, v in NET_DICT.items()]
my_pop.replace_ind_networks(names, nets)

my_optimization = SetMutRateOptimization(my_sim, my_env, my_pop, MUT_NET_PROB_DIST)
my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                    directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                    time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                    save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)

