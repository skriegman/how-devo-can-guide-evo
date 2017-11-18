from glob import glob
import cPickle
import time
import ast
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from evosoro.softbot import Genotype

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
colors = sns.color_palette("muted", 3)  # [sns.xkcd_rgb["dark pink"], sns.xkcd_rgb["cadet blue"], sns.xkcd_rgb["grey"]])
sns.set_palette(list(reversed(colors)))

RUNS = 30

USE_PICKLE = True  # make sure to change this to False if data changes

SIZE = (4, 4, 3)  # used to transform fitness into body lengths

exp_dir = "/home/sam/Archive/skriegma/AFPO_Both_M_48_4"

start_time = time.time()

if not USE_PICKLE:
    ancestor_dict = {}

    MyGenotype = Genotype

    for run in range(1, RUNS+1):
        run_directory = exp_dir + "/run_{}/".format(run)

        ancestor_dict[run] = {'id': [], 'fit': [], 'window_size': [], 'window_offset': []}

        best_of_gen_file = run_directory + "bestSoFar/bestOfGen.txt"  # use only as last resort
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
                        # if len(lineage) == 0 or this_id == lineage[-1]:
                        #     lineage += [parent_id]
                        #     ancestor_dict[run]['id'] += [int(line.split()[1])]
                        #     ancestor_dict[run]['fit'] += [float(line.split()[5])]
                        #     ancestor_dict[run]['window_size'] += [float(line.split()[28])]
                        #     ancestor_dict[run]['window_offset'] += [float(line.split()[25])]
                        #     if parent_id == -1:
                        #         first_ancestor = True
                        if len(lineage) == 0 or this_id == lineage[-1]:
                            lineage += [parent_id]
                            ancestor_dict[run]['id'] += [int(line.split()[1])]
                            ancestor_dict[run]['fit'] += [float(line.split()[5])]

                            s = ast.literal_eval(re.split(r'\t+', line)[33])
                            f = ast.literal_eval(re.split(r'\t+', line)[34])
                            s, f = np.array(s), np.array(f)
                            w = np.sum(np.abs(f-s))
                            true = float(line.split()[28])
                            assert(abs(true-w) < 10e-9)

                            s_cube = np.resize(s, (4, 4, 3))
                            reversed_array = s_cube[::-1, :, :]
                            if np.all(s_cube[:int(SIZE[0] / 2.0), :, :] == reversed_array[:int(SIZE[0] / 2.0), :, :]):
                                ancestor_dict[run]['window_size'] += [float(line.split()[28])]
                                ancestor_dict[run]['window_offset'] += [float(line.split()[25])]

                            else:
                                # print run
                                ancestor_dict[run]['window_size'] += [float(line.split()[25])]
                                ancestor_dict[run]['window_offset'] += [float(line.split()[28])]

                                s1 = ast.literal_eval(re.split(r'\t+', line)[31])
                                f1 = ast.literal_eval(re.split(r'\t+', line)[32])
                                s1, f1 = np.array(s1), np.array(f1)
                                w1 = np.sum(np.abs(f1 - s1))
                                true = float(line.split()[25])
                                assert (abs(true - w1) < 10e-9)

                                s1_cube = np.resize(s1, (4, 4, 3))
                                reversed_array1 = s1_cube[::-1, :, :]
                                if not np.all(s1_cube[:int(SIZE[0]/2.0), :, :] == reversed_array1[:int(SIZE[0]/2.0), :, :]):
                                    print "Oh shit"
                                    raise AssertionError

                            if parent_id == -1:
                                first_ancestor = True

            gens_back += 1

        run_ids = ancestor_dict[run]['id']
        run_fit = ancestor_dict[run]['fit']
        run_size_windows = ancestor_dict[run]['window_size']
        run_offset_windows = ancestor_dict[run]['window_offset']
        ancestor_dict[run]['window_Volume'] = [win for (ids, win) in sorted(zip(run_ids, run_size_windows))]
        ancestor_dict[run]['window_Phase'] = [win for (ids, win) in sorted(zip(run_ids, run_offset_windows))]
        ancestor_dict[run]['window_Fitness'] = [fit for (ids, fit) in sorted(zip(run_ids, run_fit))]
        # print sorted(zip(run_ids, run_fit))

        # print ancestor_dict[run]['window_Fitness'][-1] == pop[0].fitness

    with open('/home/sam/Projects/evosoro/evosoro/data_analysis/results/ancestor_dict.pickle', 'wb') as handle:
        cPickle.dump(ancestor_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)

    duration = time.time() - start_time
    print "cpu time: {} mins".format(duration/60.)

else:
    with open('/home/sam/Projects/evosoro/evosoro/data_analysis/results/ancestor_dict.pickle', 'rb') as handle:
        ancestor_dict = cPickle.load(handle)

# PLOTTING

max_ancestors = np.max([len(details['id']) for run, details in ancestor_dict.items()])

fig = plt.figure(figsize=(20, 20))
outer_grid = gridspec.GridSpec(3, 10, wspace=0.0, hspace=0.3)

for run in range(1, RUNS+1):
    this_num_ancestors = len(ancestor_dict[run]['id'])
    diff_from_max_len = max_ancestors - this_num_ancestors
    fill = [None] * diff_from_max_len
    vol_data = fill + ancestor_dict[run]['window_Volume']
    phase_data = fill + ancestor_dict[run]['window_Phase']
    fit_data = fill + ancestor_dict[run]['window_Fitness']
    fit_data = [fit/float(SIZE[1]) if fit is not None else fit for fit in fit_data]

    outer_idx = 0
    inner_idx = run - 1
    if run > 10:
        outer_idx = 1
        inner_idx = run - 1 - 10
    if run > 20:
        outer_idx = 2
        inner_idx = run - 1 - 20

    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 10, subplot_spec=outer_grid[outer_idx, :], wspace=0.0, hspace=0.0)
    axes0 = plt.subplot(inner_grid[0, inner_idx])
    axes1 = plt.subplot(inner_grid[1, inner_idx])
    axes2 = plt.subplot(inner_grid[2, inner_idx])

    axes0.plot(range(max_ancestors), vol_data, color=sns.color_palette()[0], linewidth=3)
    axes1.plot(range(max_ancestors), phase_data, color=sns.color_palette()[1], linewidth=3)
    axes2.plot(range(max_ancestors), fit_data, color=sns.color_palette()[2], linewidth=3)

    # for a in [axes0, axes1, axes2]:
    #     dashes = [5, 5, 5, 5]  # x points on, y off, z on, a off
    #     a.axvline(x=np.argmax(vol_data), color='grey', linewidth=1, dashes=dashes, zorder=0, alpha=1)

    for ax in [axes0, axes1, axes2]:
        ax.set_xlim([0, max_ancestors+1])
        ax.set_xticks([max_ancestors / 2.])
        ax.set_xticklabels([])

    for ax in [axes0, axes1]:
        ax.set_ylim([0, .7*96])  # devo
        ax.set_yticks([])

    axes2.set_ylim([0, 80])  # fitness
    axes2.set_yticks([])

    ax2 = axes0.twiny()
    ax2.set_title('run {}'.format(run), )
    ax2.set_xticks([])

    ticks = np.array([.1, .3, .5])
    if inner_idx == 0:
        for ax in [axes0, axes1]:
            ax.set_yticks(ticks*96)
            ax.yaxis.set_ticklabels(ticks)

        axes2.set_yticks([20, 40, 60])
        axes0.set_ylabel(r"$\mathregular{W_{L}}$", fontsize=25)
        axes1.set_ylabel(r"$\mathregular{W_{\Phi}}$", fontsize=25)
        axes2.set_ylabel("$\mathregular{F}$", fontsize=25)
        axes2.set_xlabel("$\mathregular{T}$", fontsize=25)
        axes2.set_xticklabels(["old $\\rightarrow$ new"])

    if outer_idx == 0 and inner_idx == 4:
        axes0.set_title("          Developmental Windows\n", fontsize=35)

    if (outer_idx == 0 and inner_idx in [5, 7]) or (outer_idx == 1 and inner_idx in [5, 6, 7]):
        axes2.set_xticklabels(["$^{\mathregular{\\ast}}$"], fontsize=55)


plt.savefig("plots/Journal_Devo_Windows.tiff", bbox_inches='tight', format='tiff', dpi=4500/20)
