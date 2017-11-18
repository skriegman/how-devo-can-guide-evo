import numpy as np
from glob import glob
import subprocess as sub
import ast
import time


NAME = "_Birth"
LAMBDAS = ["M_48_4"]  # [1, 2, 4, 8, 16, 24, 32, 48]
SIZE = (4, 4, 3)
RUNS = 30
FREEZE_CONTROLLER = False
FREEZE_MORPHOLOGY = True
PAUSE_BETWEEN_LAMBDA = False


def freeze_stat(birth, death):
    """Replaces two genomes (birth, death) with a single frozen genome"""
    # return (np.array(birth) + np.array(death)) * 0.5
    return np.array(birth)


def print_layer(layer):
    layer = str(list(layer))
    layer = layer[:len(layer)-1] + ", ]"
    return "<Layer><![CDATA{}]></Layer> \n".format(layer)


def get_data_from_layer(layer):
    return layer[layer.find("<Layer><![CDATA[") + len("<Layer><![CDATA["):layer.find("]]></Layer>")]


for LAMBDA in LAMBDAS:
    for run in range(1, RUNS+1):
        run_dir = "/home/sam/Archive/skriegma/AFPO_Both_{0}/run_{1}/bestSoFar".format(LAMBDA, run)
        best_of_gen_files = glob(run_dir + "/fitOnly/*")
        run_champ = sorted(best_of_gen_files, reverse=True)[0]
        # print run_champ
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
            if size_count > SIZE[2] + 1:
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
            if phase_count > SIZE[2] + 1:
                phase_count = 0
                final = False

        f.close()
        midlife_size = freeze_stat(init_size, final_size)
        midlife_phase = freeze_stat(init_phase, final_phase)

        frozen_run_champ = run_dir + "/Frozen_Run_Champ{}.vxa".format(NAME)
        f = open(frozen_run_champ, "w")

        size_count = 0
        phase_count = 0
        final = False
        for line in lines:
            wrote = False
            if "Final" in line:
                final = True

            if "Size" in line:
                size_count += 1
            elif size_count > 0:
                size_count += 1
                if not final and FREEZE_MORPHOLOGY:
                    f.write(print_layer(midlife_size[size_count-2]))
                    wrote = True

            if "Phase" in line:
                phase_count += 1
            elif phase_count > 0:
                phase_count += 1
                if not final and FREEZE_CONTROLLER:
                    f.write(print_layer(midlife_phase[phase_count-2]))
                    wrote = True

            if not wrote and not final and (phase_count == 0 or size_count == 0) and "fitness" not in line:
                f.write(line)
                wrote = True

            if "fitness" in line:
                f.write('        <FitnessFileName>{}</FitnessFileName>\n'.format(run_dir + "/Frozen_Fitness{}.xml".format(NAME)))
                wrote = True

            if not wrote:
                if not FREEZE_MORPHOLOGY and size_count > 0:
                    f.write(line)
                    wrote = True
                elif not FREEZE_CONTROLLER and phase_count > 0:
                    f.write(line)
                    wrote = True

            if phase_count > SIZE[2] + 1:
                phase_count = 0
                final = False

            if size_count > SIZE[2] + 1:
                size_count = 0
                final = False

        f.close()

        # re-evaluate with physics engine
        sub.Popen("../_voxcad/voxelyzeMain/voxelyze .  -f " + frozen_run_champ, shell=True)

    if PAUSE_BETWEEN_LAMBDA:
        time.sleep(105)

