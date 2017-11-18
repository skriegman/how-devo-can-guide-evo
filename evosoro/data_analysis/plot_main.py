import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman', 'axes.edgecolor': '0.01'})
sns.set_palette("Set2", 8)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

########################################################################################################################
df = pd.read_csv('results/best_results_co_opt_M_48_4.csv', low_memory=False)
df = df.fillna(0)
df.ix[df.fitness <= 0, 'fitness'] = 0

df['fitness'] *= 0.25  # body lengths

orig_len = len(df)
print len(df[df.duplicated(subset=["id", "condition", "gen", "run"])]) == \
      len(df[df.duplicated(subset=["id", "condition", "gen", "run", "fitness"])])
print len(df[df.duplicated(subset=["id", "condition", "gen", "run"])]), " duplicates"
df = df.drop_duplicates(subset=["id", "condition", "gen", "run"], keep='last')

b = sns.tsplot(data=df, value="fitness", condition="condition", unit="run", time="gen",
               ax=axes[0],
               ci=95,
               estimator=np.median,
               legend=False
               )

axes[0].set_ylim([0, 60])
axes[0].set_xlim([-50, 10000])
axes[0].set_xticks(range(0, 10001, 2000))
axes[0].set_yticks(range(0, 61, 10))
axes[0].set_ylabel("Body lengths traveled")
axes[0].set_xlabel("Generation")
axes[0].legend([mpatches.Patch(color=sns.color_palette()[i]) for i in range(2)], ['Evo', 'Evo-Devo'], loc=2)

########################################################################################################################

NAMES = ["_Birth", "_Midlife", "_Death",
         "_Control_Birth", "_Control_Midlife", "_Control_Death",
         "_Body_Birth", "_Body_Midlife", "_Body_Death"]
LAMBDAS = ["M_48_4"]
RUNS = 30


def get_control_fit(lamb, run):
    run_dir = "/home/sam/Archive/skriegma/AFPO_None_{0}/run_{1}/bestSoFar".format(lamb, run)
    best_of_gen_file = run_dir + "/bestOfGen.txt"
    with open(best_of_gen_file, 'r') as infile:
        lines = infile.readlines()
        return float(lines[-1].split()[5])


def get_plastic_fit(lamb, run):
    run_dir = "/home/sam/Archive/skriegma/AFPO_Both_{0}/run_{1}/bestSoFar".format(lamb, run)
    best_of_gen_file = run_dir + "/bestOfGen.txt"
    with open(best_of_gen_file, 'r') as infile:
        lines = infile.readlines()
        return float(lines[-1].split()[5])


control_fitness_dict = {lamb: {run: 0 for run in range(1, RUNS+1)} for lamb in LAMBDAS}
plastic_fitness_dict = {lamb: {run: 0 for run in range(1, RUNS+1)} for lamb in LAMBDAS}
frozen_fitness_dict = {name: {lamb: {run: 0 for run in range(1, RUNS+1)} for lamb in LAMBDAS} for name in NAMES}
for name in NAMES:
    for LAMBDA in LAMBDAS:
        for run in range(1, RUNS+1):
            control_fitness_dict[LAMBDA][run] = get_control_fit(LAMBDA, run)
            plastic_fitness_dict[LAMBDA][run] = get_plastic_fit(LAMBDA, run)
            run_dir = "/home/sam/Archive/skriegma/AFPO_Both_{0}/run_{1}/bestSoFar".format(LAMBDA, run)
            this_file = open(run_dir + "/Frozen_Fitness{}.xml".format(name))
            tag = '<finalDistY>'
            for line in this_file:
                if tag in line:
                    frozen_fitness_dict[name][LAMBDA][run] = float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])


# print control_fitness_dict
# print plastic_fitness_dict
# print frozen_fitness_dict['_Birth']
# print frozen_fitness_dict['_Midlife']
# print frozen_fitness_dict['_Death']

control = []
orig = []
m = []
s = []
f = []
sc = []
mc = []
fc = []
for lamb in LAMBDAS:
    control_lamb = []
    orig_lamb = []
    m_lamb = []
    s_lamb = []
    f_lamb = []
    sc_lamb = []
    mc_lamb = []
    fc_lamb = []
    for run in range(1, RUNS+1):
        control_lamb += [control_fitness_dict[lamb][run]]
        orig_lamb += [plastic_fitness_dict[lamb][run]]
        m_lamb += [frozen_fitness_dict[NAMES[0]][lamb][run]]
        s_lamb += [frozen_fitness_dict[NAMES[1]][lamb][run]]
        f_lamb += [frozen_fitness_dict[NAMES[2]][lamb][run]]
        sc_lamb += [frozen_fitness_dict[NAMES[3]][lamb][run]]
        mc_lamb += [frozen_fitness_dict[NAMES[4]][lamb][run]]
        fc_lamb += [frozen_fitness_dict[NAMES[5]][lamb][run]]

    control += [control_lamb]
    orig += [orig_lamb]
    m += [m_lamb]
    s += [s_lamb]
    f += [f_lamb]
    sc += [sc_lamb]
    mc += [mc_lamb]
    fc += [fc_lamb]

control = np.array(control) / 4.
orig = np.array(orig) / 4.
m = np.array(m) / 4.
s = np.array(s) / 4.
f = np.array(f) / 4.
sc = np.array(sc) / 4.
mc = np.array(mc) / 4.
fc = np.array(fc) / 4.

orig_balls = orig >= 10
print "orig balls: ", np.sum(orig_balls)
no_m_balls = m < 10
no_s_balls = s < 10
no_f_balls = f < 10
no_sc_balls = sc < 16
no_mc_balls = mc < 16
no_fc_balls = fc < 16

fitness = np.array([x[0] for x in [control, orig, s]])  # m, s, f, sc, mc, fc]])
fitness = fitness.flatten()

runs = np.array([range(1, RUNS+1) for x in [control, orig, s]])  # m, s, f, sc, mc, fc]])
runs.flatten()

group = np.array([[i]*30 for i, n in enumerate(["Evo", "Evo-Devo", "birth"])])  # "mid", "birth", "death", "cmid", "cbirth", "cdeath"])])
group = group.flatten()


data = np.array([group, fitness])
# print data

df = pd.DataFrame(data=data.T, columns=["Group", "Fitness"])

g = sns.barplot(x="Group", y="Fitness", data=df, estimator=np.median, ax=axes[1], capsize=0.1, errwidth=2)

axes[1].set_ylabel("Body lengths traveled")
# axes[1].set_xlabel("~")
axes[1].xaxis.label.set_visible(False)
axes[1].set_xticklabels(["Evo", "Evo-Devo", "Evo-Devo\nremoved"])

# ax.set_yticklabels(range(0, 61, 10))
plt.ylim([0, 60])

# fig.text(0.598, 0.079, "      Evo            Evo-Devo       Evo-Devo\n "
#                        "                                               at birth",
#          fontsize=16, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1, 'edgecolor': 'white'})


axes[0].text(axes[0].get_xlim()[1], axes[0].get_ylim()[1], "A", ha="right", va="center", fontsize=30, fontname="Arial")
axes[1].text(axes[1].get_xlim()[1], axes[1].get_ylim()[1], "B", ha="right", va="center", fontsize=30, fontname="Arial")

sns.despine()
plt.tight_layout()
plt.savefig("plots/Journal_Main_Results.tiff", bbox_inches='tight', format='tiff', dpi=4500/12)

