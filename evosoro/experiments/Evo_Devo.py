import random
import os
import sys
import numpy as np
import subprocess as sub

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import DirectEncoding
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import SetMutRateOptimization
from evosoro.tools.checkpointing import continue_from_checkpoint

sub.call("cp ~/pkg/evosoro/evosoro/_voxcad/voxelyzeMain/voxelyze .", shell=True)

SEED = int(sys.argv[1])
MAX_TIME = float(sys.argv[2])

POP_SIZE = 30
IND_SIZE = (4, 4, 3)
MAX_GENS = 10000
NUM_RANDOM_INDS = 1

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

TIME_TO_TRY_AGAIN = 12
MAX_EVAL_TIME = 30

SAVE_VXA_EVERY = MAX_GENS+1  # never save
SAVE_LINEAGES = False
CHECKPOINT_EVERY = 100
EXTRA_GENS = 0

RUN_DIR = "run_{}".format(SEED)
RUN_NAME = "Both"


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


if not os.path.isfile("./" + RUN_DIR + "/pickledPops/Gen_0.pickle"):

    random.seed(SEED)
    np.random.seed(SEED)

    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT,
                 fitness_eval_init_time=INIT_TIME)

    my_env = Env(temp_amp=TEMP_AMP)
    my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")

    my_objective_dict = ObjectiveDict()
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<finalDistY>")
    my_objective_dict.add_objective(name="age", maximize=False, tag=None)

    my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POP_SIZE)

    my_optimization = SetMutRateOptimization(my_sim, my_env, my_pop, MUT_NET_PROB_DIST)
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)

else:
    continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                             max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                             checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_VXA_EVERY,
                             save_lineages=SAVE_LINEAGES)
