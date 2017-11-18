from glob import glob
import cPickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns

from evosoro.softbot import Genotype

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
sns.set_palette("Set2", 8)
# sns.set_palette([sns.color_palette()[0], sns.color_palette()[2]])


USE_PICKLE = True  # make sure to change this to False if data changes
RUNS = 10
POP_SIZE = 30
SIZE = (4, 4, 3)  # used to transform fitness into body lengths

path = "/home/sam/Archive/skriegma/"
EXP_NAMES = ["Devo", "Evo"]

start_time = time.time()

f, axes = plt.subplots(2, 2, figsize=(14, 8))

names = ["A", "B", "C", "D"]

n = -1
for MORPHO in [0, 1]:
    for t in [0, 1]:
        n += 1

        if not USE_PICKLE:
            ancestor_dict = {name: {} for name in EXP_NAMES}

            MyGenotype = Genotype

            for exp in EXP_NAMES:
                exp_dir = path+"Random_{0}_Walks{1}".format(exp, "_Morpho" if MORPHO else "")

                for run in range(1, RUNS+1):
                    print "getting run: {}".format(run)
                    ancestor_dict[exp][run] = {}

                    for idx in range(POP_SIZE):
                        run_directory = exp_dir + "/run_{}/".format(run)

                        ancestor_dict[exp][run][idx] = {'id': [], 'fit': []}

                        all_gen_files = glob(run_directory + "allIndividualsData/Gen_*.txt")
                        sorted_gen_files = sorted(all_gen_files, reverse=True)

                        first_ancestor = False
                        run_champ_id, run_champ_fitness = -1, -1
                        lineage = []
                        gens_back = 0
                        while not first_ancestor:

                            with open(sorted_gen_files[gens_back], 'r') as infile:
                                count = -1
                                for line in infile:
                                    write = False
                                    if line.split()[0] != 'gen':
                                        this_id = int(line.split()[1])
                                        parent_id = int(line.split()[3])

                                        if len(lineage) == 0:
                                            if count == idx:
                                                write = True
                                        elif this_id == lineage[-1]:
                                            write = True

                                        if write:
                                            lineage += [parent_id]
                                            ancestor_dict[exp][run][idx]['id'] += [int(line.split()[1])]
                                            ancestor_dict[exp][run][idx]['fit'] += [float(line.split()[5])]

                                            if parent_id == -1:
                                                first_ancestor = True
                                    count += 1

                            gens_back += 1

                        run_ids = ancestor_dict[exp][run][idx]['id']
                        run_fit = ancestor_dict[exp][run][idx]['fit']
                        ancestor_dict[exp][run][idx]['Sorted_Fitness'] = [fit for (ids, fit) in sorted(zip(run_ids, run_fit))]

            with open('/home/sam/Projects/evosoro/evosoro/data_analysis/results/random_walks_dict{}.pickle'.format("_morpho" if MORPHO else ""), 'wb') as handle:
                cPickle.dump(ancestor_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)

            duration = time.time() - start_time
            print "cpu time: {} mins".format(duration/60.)

        else:
            with open('/home/sam/Projects/evosoro/evosoro/data_analysis/results/random_walks_dict{}.pickle'.format("_morpho" if MORPHO else ""), 'rb') as handle:
                ancestor_dict = cPickle.load(handle)

        if t == 0:
            Evo = {k: [] for k in range(POP_SIZE)}
            EvoDevo = {k: [] for k in range(POP_SIZE)}

            for ind in range(POP_SIZE):
                for run in range(1, RUNS+1):
                    this_devo_id = ancestor_dict["Devo"][run][ind]["id"][-1]
                    orig_devo_fit = ancestor_dict["Devo"][run][ind]["fit"][-1]
                    this_devo_fit = ancestor_dict["Devo"][run][ind]["fit"]
                    EvoDevo[this_devo_id] += [[x/orig_devo_fit for x in this_devo_fit]]

                    this_evo_id = ancestor_dict["Evo"][run][ind]["id"][-1]
                    orig_evo_fit = ancestor_dict["Evo"][run][ind]["fit"][-1]
                    this_evo_fit = ancestor_dict["Evo"][run][ind]["fit"]
                    Evo[this_evo_id] += [[x / orig_evo_fit for x in this_evo_fit]]

            # Evo = [np.median(v, 0) for k, v in Evo.items() if len(v) > 0]
            # EvoDevo = [np.median(v, 0) for k, v in EvoDevo.items() if len(v) > 0]
            Evo = [v for k, v in Evo.items() if len(v) > 0]
            EvoDevo = [v for k, v in EvoDevo.items() if len(v) > 0]

            EvoDevo, Evo = np.array(EvoDevo), np.array(Evo)
            EvoDevo, Evo = EvoDevo[:, ::-1], Evo[:, ::-1]

            Evo, EvoDevo = Evo.flatten()[::-1], EvoDevo.flatten()[::-1]

            delta = np.concatenate([Evo, EvoDevo])
            condition = ["Evo" for x in Evo] + ["EvoDevo" for x in EvoDevo]

            step = np.arange(1.05, 1002.05)
            step = np.tile(step, 2*30*RUNS)

            run = [r for r in range(30*RUNS)]
            run = np.repeat(run, 1001)
            run = np.tile(run, 2)

            print len(delta), len(condition), len(run), len(step)

            df = pd.DataFrame({'run': run, 'step': step, 'delta': delta, 'condition': condition})

            sns.set_palette("Set2", 8)
            b = sns.tsplot(data=df, value="delta", condition="condition", unit="run", time="step",
                           estimator=np.median,
                           ci=99,
                           legend=False,
                           ax=axes[MORPHO, t]
                           )


        else:
            Fast = {k: [] for k in range(POP_SIZE)}
            Slow = {k: [] for k in range(POP_SIZE)}

            for ind in range(POP_SIZE):
                for run_b in range(1, RUNS + 1):
                    this_id = ancestor_dict["Devo"][run_b][ind]["id"][-1]
                    orig_fit = ancestor_dict["Devo"][run_b][ind]["fit"][-1]
                    this_fit = ancestor_dict["Devo"][run_b][ind]["fit"]
                    if ancestor_dict["Devo"][run_b][ind]["id"][-1] in [5, 7, 15, 16, 17]:
                        Slow[this_id] += [[x / orig_fit for x in this_fit]]
                    else:
                        Fast[this_id] += [[x / orig_fit for x in this_fit]]

            # Slow = [np.median(v, 0) for k, v in Slow.items() if len(v) > 0]
            # Fast = [np.median(v, 0) for k, v in Fast.items() if len(v) > 0]
            Slow = [v for k, v in Slow.items() if len(v) > 0]
            Fast = [v for k, v in Fast.items() if len(v) > 0]

            Fast, Slow = np.array(Fast), np.array(Slow)
            Fast, Slow = Fast[::-1], Slow[::-1]

            Fast, Slow = Fast.flatten()[::-1], Slow.flatten()[::-1]

            delta_b = np.concatenate([Fast, Slow])
            condition_b = ["Fast" for x in Fast] + ["Slow" for x in Slow]

            step_b = np.arange(1.05, 1002.05)
            step_b = np.tile(step_b, 30*RUNS)

            run_b = [6, 8, 16, 17, 18]
            runf_b = [r for r in range(1, 31, 1) if r not in run_b]

            run_b = np.repeat(range(len(run_b*RUNS)), 1001)
            runf_b = np.repeat(range(len(run_b*RUNS), len(run_b*RUNS) + len(runf_b*RUNS)), 1001)

            run_b = np.concatenate([run_b, runf_b])

            print len(delta_b), len(condition_b), len(run_b), len(step_b)

            df_b = pd.DataFrame({'run': run_b, 'step': step_b, 'delta': delta_b, 'condition': condition_b})

            sns.set_palette("Set2", 8)
            sns.set_palette(sns.color_palette()[3:])
            b = sns.tsplot(data=df_b, value="delta", condition="condition", unit="run", time="step",
                           estimator=np.median,
                           ci=99,
                           legend=False,
                           ax=axes[MORPHO, t]
                           )

        axes[MORPHO, t].set_ylim([-0.025, 1.0])
        axes[MORPHO, t].set_xlim([1, None])

        axes[MORPHO, t].set_xscale("log")

        axes[MORPHO, t].set_ylabel("Relative fitness")
        axes[MORPHO, t].set_xlabel("Random steps in {} space".format("control" if not MORPHO else "morpho"))

        if t == 0:
            sns.set_palette("Set2", 8)
            axes[MORPHO, t].legend([mpatches.Patch(color=sns.color_palette()[i]) for i in range(2)], ['Evo', 'Evo-Devo'], loc=(0.66, 0.3))
        else:
            sns.set_palette("Set2", 8)
            sns.set_palette(sns.color_palette()[3:])
            axes[MORPHO, t].legend([mpatches.Patch(color=sns.color_palette()[i]) for i in range(2)], ['Evo-Devo Fast', 'Evo-Devo Slow'], loc=(0.55, 0.3))

        axes[MORPHO, t].text(5, 0.7, names[n], fontsize=30, fontname="Arial")

sns.despine()
plt.tight_layout()
plt.savefig("plots/Journal_Random_Walks.tiff", bbox_inches='tight', format='tiff', dpi=4500/14)

