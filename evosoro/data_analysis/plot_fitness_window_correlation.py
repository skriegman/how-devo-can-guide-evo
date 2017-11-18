from glob import glob
import cPickle
import time
import ast
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from evosoro.softbot import Genotype

MIN_BALL_FIT = -1
CMAP = "jet"
COLOR_LIM = 60
GRID_SIZE = 30
LAMBDAS = ["M_48_4"]  # reversed([1, 2, 4, 8, 16, 24, 32, 48])
RUNS = 30
USE_PICKLE = True  # make sure to change this to False if data changes
SIZE = (4, 4, 3)  # used to transform fitness into body lengths

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
colors = sns.color_palette("muted", 3)
sns.set_palette(list(reversed(colors)))

start_time = time.time()

for mut_rate in LAMBDAS:
    print "Using mut_rate: {}".format(mut_rate)
    if not USE_PICKLE:
        overall_window_dict = {"gen": [], "id": [], "fit": [], "window_Morphology": [], "window_Controller": []}

        MyGenotype = Genotype

        for run in range(1, RUNS+1):
            print "lamb {0}: run {1} of {2}".format(mut_rate, run, RUNS)
            run_directory = "/home/sam/Archive/skriegma/AFPO_Both_{0}/run_{1}/".format(mut_rate, run)
            all_gen_files = glob(run_directory + "allIndividualsData/Gen_*.txt")

            for filename in all_gen_files:
                with open(filename, 'r') as infile:
                    n = 0
                    for line in infile:
                        if n > 0:
                            overall_window_dict['gen'] += [int(line.split()[0])]
                            overall_window_dict['id'] += [int(line.split()[1])]
                            overall_window_dict['fit'] += [float(line.split()[5])/float(SIZE[0])]
                            # overall_window_dict['window_Morphology'] += [float(line.split()[28])]
                            # overall_window_dict['window_Controller'] += [float(line.split()[25])]

                            s = ast.literal_eval(re.split(r'\t+', line)[33])
                            f = ast.literal_eval(re.split(r'\t+', line)[34])
                            s, f = np.array(s), np.array(f)
                            w = np.sum(np.abs(f - s))
                            true = float(line.split()[28])
                            assert (abs(true - w) < 10e-9)

                            s_cube = np.resize(s, (4, 4, 3))
                            reversed_array = s_cube[::-1, :, :]
                            if np.all(s_cube[:int(SIZE[0] / 2.0), :, :] == reversed_array[:int(SIZE[0] / 2.0), :, :]):
                                overall_window_dict['window_Morphology'] += [float(line.split()[28])]
                                overall_window_dict['window_Controller'] += [float(line.split()[25])]

                            else:
                                # print run
                                overall_window_dict['window_Morphology'] += [float(line.split()[25])]
                                overall_window_dict['window_Controller'] += [float(line.split()[28])]

                                s1 = ast.literal_eval(re.split(r'\t+', line)[31])
                                f1 = ast.literal_eval(re.split(r'\t+', line)[32])
                                s1, f1 = np.array(s1), np.array(f1)
                                w1 = np.sum(np.abs(f1 - s1))
                                true = float(line.split()[25])
                                assert (abs(true - w1) < 10e-9)

                                s1_cube = np.resize(s1, (4, 4, 3))
                                reversed_array1 = s1_cube[::-1, :, :]
                                if not np.all(s1_cube[:int(SIZE[0]/2.0), :, :] == reversed_array1[:int(SIZE[0]/2.0), :, :]):
                                    raise AssertionError("corrupt data!")

                        n += 1

        print "PICKLING..."
        with open('/home/sam/Projects/evosoro/evosoro/data_analysis/results/overall_window_dict_{}.pickle'.format(mut_rate), 'wb') as handle:
            cPickle.dump(overall_window_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)

        duration = time.time() - start_time
        print "{0}  cpu time: {1} mins".format(mut_rate, duration/60.)

    else:
        with open('/home/sam/Projects/evosoro/evosoro/data_analysis/results/overall_window_dict_{}.pickle'.format(mut_rate), 'rb') as handle:
            overall_window_dict = cPickle.load(handle)

    # PLOTTING
    print "PLOTTING..."

    title = "$\lambda=${}/48".format(mut_rate)
    ball_idx = np.array(overall_window_dict['fit']) > MIN_BALL_FIT
    # print np.sum(ball_idx)

    # for color, name in enumerate(["Morphology", "Controller"]):
    #     f, axes = plt.subplots(1, 1, figsize=(4, 4))
    #
    #     fit = np.array(overall_window_dict['fit'])
    #     window = np.array(overall_window_dict['window_{}'.format(name)])
    #
    #     g = sns.jointplot(fit[ball_idx], window[ball_idx],
    #                       kind="hex", xlim=(0, 80), ylim=(0, 96),
    #                       color=sns.color_palette()[color],
    #                       marginal_kws={"bins": range(0, 96, 2)},
    #                       joint_kws={"gridsize": GRID_SIZE, "extent": (MIN_BALL_FIT, 96, 0, 80)},
    #                       stat_func=None,
    #                       linewidths=0.01
    #                       ).set_axis_labels("Fitness", "{} Window".format(name))
    #
    #     x_start = MIN_BALL_FIT
    #     # g.ax_joint.fill_between(np.arange(0, x_start, 0.01), 0, g.ax_joint.get_ylim()[1], facecolor="lightgrey")
    #     # g.ax_marg_x.fill_between(np.arange(0, x_start, 0.01), 0, g.ax_marg_x.get_ylim()[1], facecolor="lightgrey")
    #
    #     g.ax_joint.set_yticks(np.arange(0, 96+1, 96/5.))
    #     g.ax_joint.yaxis.set_ticklabels(np.arange(0, 1.1, 1/5.))
    #
    #     # g.ax_joint.annotate(title, xy=(55, 57.5))
    #
    #     plt.tight_layout()
    #     plt.savefig("plots/Fitness_Correlation_{0}_{1}.pdf".format(name, mut_rate), bbox_inches='tight', transparent=True)
    #     plt.clf()
    #     plt.close()
    #

    f, axes = plt.subplots(1, 1, figsize=(6, 5))

    fitness = np.array(overall_window_dict['fit'])
    window_c = np.array(overall_window_dict['window_Controller'])
    window_m = np.array(overall_window_dict['window_Morphology'])
    fitness = np.nan_to_num(fitness)
    last_gen_idx = np.array(overall_window_dict['gen']) == np.max(overall_window_dict['gen'])
    # real_idx = np.logical_and(fitness != np.nan, fitness >= -0.1)
    fit, win_c, win_m = fitness[ball_idx], window_c[ball_idx], window_m[ball_idx]
    # fit, win_c, win_m = fitness, window_c, window_m

    plt.hexbin(win_c, win_m, C=fit,
               gridsize=GRID_SIZE,
               extent=(0, 100, 0, 100),
               cmap=CMAP, linewidths=0.01,
               reduce_C_function=np.median,
               vmin=MIN_BALL_FIT)

    # axes.set_title("Median fitness".format(mut_rate), fontsize=15)
    axes.set_ylabel("Morphological development", fontsize=15)
    axes.set_xlabel("Controller development", fontsize=15)
    # axes.set_ylabel("Morphological development ($\mathregular{W_{L}}$)", fontsize=15)
    # axes.set_xlabel("Controller development ($\mathregular{W_{\Phi}}$)", fontsize=15)
    # axes.set_ylim([0, 72])
    # axes.set_xlim([0, 72])
    # axes.set_yticks(np.arange(0, 72+1, 72/8.))
    # axes.set_xticks(np.arange(0, 72+1, 72/8.))
    # axes.xaxis.set_ticklabels(np.arange(0, 0.81, .1))
    # axes.yaxis.set_ticklabels(np.arange(0, 0.81, .1))
    axes.set_ylim([0, 96])
    axes.set_xlim([0, 96])
    axes.set_yticks(np.arange(0, 96+1, 96/5.))
    axes.set_xticks(np.arange(0, 96+1, 96/5.))
    axes.xaxis.set_ticklabels(np.arange(0, 1.01, 1/5.), fontsize=15)
    axes.yaxis.set_ticklabels(np.arange(0, 1.01, 1/5.), fontsize=15)

    cb = plt.colorbar(ticks=range(MIN_BALL_FIT+1, COLOR_LIM+1, 10), boundaries=range(MIN_BALL_FIT, COLOR_LIM+1))
    cb.set_clim(MIN_BALL_FIT, COLOR_LIM)
    cb.ax.tick_params(labelsize=15)

    f.text(0.95, 0.8965, "fitness", fontsize=15, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})
    f.text(0.9645, 0.8965 + .042, "High", fontsize=15,  bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})
    f.text(0.95, 0.1425, "fitness", fontsize=15, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})
    f.text(0.971, 0.1425 + 0.042, "Low", fontsize=15, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})

    plt.tight_layout()
    plt.savefig("plots/Honeycomb.tiff".format(mut_rate, MIN_BALL_FIT), bbox_inches='tight', format='tiff', dpi=5.2*600/6)
    plt.clf()
    plt.close()

