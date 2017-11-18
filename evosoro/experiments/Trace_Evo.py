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
from evosoro.tools.utils import reorder_vxa_array
from evosoro.tools.data_analysis import get_data_from_layer


sub.call("cp ../_voxcad/voxelyzeMain/voxelyze .", shell=True)


SEED = int(sys.argv[1])
MAX_TIME = float(sys.argv[2])

POP_SIZE = 30  # defined below based on length of lineage
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

MUT_NET_PROB_DIST = [1, 0.5, 0.5]  # first prob is for meta mutation
MUT_RATE = 1/48.  # starting and lower bound of meta mutation rate (one voxel)
MUT_SCALE = 1
META_MUT_RATE = 48/48.
META_MUT_SCALE = 4/48.

STOP_AFTER_FALLING = False

NET_DICT = {"init_size": [], "init_offset": []}

TIME_TO_TRY_AGAIN = 12
MAX_EVAL_TIME = 30

SAVE_TRACES = True
TIME_BETWEEN_TRACES = 0.01  # in sec

SAVE_VXA_EVERY = MAX_GENS+1  # never save
SAVE_LINEAGES = False
CHECKPOINT_EVERY = 100
EXTRA_GENS = 0

RUN_DIR = "run_{}".format(SEED)
RUN_NAME = "Traces"

for run in range(1, 31):
    run_dir = "/home/sam/Archive/skriegma/AFPO_None_M_48_4/run_{}/bestSoFar".format(run)
    best_of_gen_files = glob(run_dir + "/fitOnly/*")
    run_champ = sorted(best_of_gen_files, reverse=True)[0]
    run_champ_clone = run_dir + "/Run_Champ.vxa"
    sub.call("cp " + run_champ + " " + run_champ_clone, shell=True)

    init_size = []
    final_size = []
    init_phase = []
    final_phase = []

    f = open(run_champ_clone, 'r')
    lines = f.readlines()

    size_count = 0
    phase_count = 0
    final = False
    for line in lines:
        if "Final" in line:
            final = True

        if "Size" in line:
            size_count += 1
        elif size_count > 0:
            size_count += 1
            if final:
                string = get_data_from_layer(line)
                final_size.append(ast.literal_eval(string))
            else:
                string = get_data_from_layer(line)
                init_size.append(ast.literal_eval(string))
        if size_count > IND_SIZE[2] + 1:
            size_count = 0
            final = False

        if "Phase" in line:
            phase_count += 1
        elif phase_count > 0:
            phase_count += 1
            if final:
                string = get_data_from_layer(line)
                final_phase.append(ast.literal_eval(string))
            else:
                string = get_data_from_layer(line)
                init_phase.append(ast.literal_eval(string))
        if phase_count > IND_SIZE[2] + 1:
            phase_count = 0
            final = False

    f.close()

    init_size, init_phase = np.array(init_size), np.array(init_phase)

    midlife_size = reorder_vxa_array(init_size, IND_SIZE)
    midlife_phase = reorder_vxa_array(init_phase, IND_SIZE)

    NET_DICT["init_size"] += [midlife_size]
    NET_DICT["init_offset"] += [midlife_phase]


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(DirectEncoding(output_node_name="mutation_rate", orig_size_xyz=IND_SIZE,
                                        scale=META_MUT_SCALE, p=META_MUT_RATE, symmetric=False,
                                        lower_bound=MUT_RATE, start_val=MUT_RATE, mutate_start_val=True))

        self.add_network(DirectEncoding(output_node_name="init_size", orig_size_xyz=IND_SIZE,
                                        scale=MUT_SCALE))
        self.to_phenotype_mapping.add_map(name="init_size", tag="<InitialVoxelSize>")

        self.add_network(DirectEncoding(output_node_name="init_offset", orig_size_xyz=IND_SIZE,
                                        scale=MUT_SCALE, symmetric=False))
        self.to_phenotype_mapping.add_map(name="init_offset", tag="<PhaseOffset>")


random.seed(SEED)
np.random.seed(SEED)

my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT,
             fitness_eval_init_time=INIT_TIME)

my_env = Env(temp_amp=TEMP_AMP, falling_prohibited=STOP_AFTER_FALLING, time_between_traces=TIME_BETWEEN_TRACES)
my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")
my_env.add_param("save_traces", int(SAVE_TRACES), "<SaveTraces>")

my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<finalDistY>")
my_objective_dict.add_objective(name="age", maximize=False, tag=None)
my_objective_dict.add_objective(name="lifetime", maximize=True, tag="<Lifetime>", logging_only=True)
my_objective_dict.add_objective(name="trace", maximize=True, tag="<CMTrace>", logging_only=True)

my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POP_SIZE)

names = [k for k, v in NET_DICT.items()]
nets = [v for k, v in NET_DICT.items()]
my_pop.replace_ind_networks(names, nets)

my_optimization = SetMutRateOptimization(my_sim, my_env, my_pop, MUT_NET_PROB_DIST)
my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                    directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                    time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                    save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)

