import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
sns.set_palette('Set2', 8)  # '[sns.xkcd_rgb["dark pink"], sns.xkcd_rgb["cadet blue"], sns.xkcd_rgb["grey"]])
sns.set_palette(sns.color_palette()[1:])

RUNS = 30
RUN_DIR = "/home/sam/Archive/skriegma/Ball_Discovery/"

lens = []
xmax = 0
for run in range(1, RUNS+1):
    if run not in [6, 8, 16, 17, 18]:
        first_gen_stats = pd.read_table(RUN_DIR + 'run_{}/allIndividualsData/Gen_0000.txt'.format(run),
                                        delimiter="\t\t", lineterminator='\n', engine='python')
        this_len = len(first_gen_stats['id'].values.tolist())
        lens += [this_len]
        if xmax < this_len:
            xmax = this_len

print np.median(lens)

discovery_time_begin = []
discovery_time_end = []
lifetimes = []
indices = []
for run in range(1, RUNS + 1):
    if run not in [6, 8, 16, 17, 18]:
        first_gen_stats = pd.read_table(RUN_DIR + 'run_{}/allIndividualsData/Gen_0000.txt'.format(run),
                                        delimiter="\t\t", lineterminator='\n', engine='python')

        lifetime = first_gen_stats['lifetime'].values.tolist()
        idx = first_gen_stats['id'].values.tolist()
        lifetime = [x-0.5 for _, x in sorted(zip(idx, lifetime), reverse=True)]

        t = 0
        found = False
        while not found:
            if lifetime[t] < 10:
                found = True
                discovery_time_begin += [lifetime[t]]
                discovery_time_end += [lifetime[-1]]
            t += 1

        diff = xmax - len(idx)
        fill = [0.0] * diff
        lifetime += fill
        idx += fill

        lifetimes += [lifetime]
        indices += [sorted(idx, reverse=True)]

lifetimes = np.array(lifetimes).flatten()
indices = np.tile(np.arange(xmax), 25)
run = np.repeat(np.arange(25), xmax)
df = pd.DataFrame({'run': run, 'lifetimes': lifetimes, 'indices': indices})


f, ax = plt.subplots(1, 2, figsize=(12, 4))


def my_estimator(x, axis):
    y = np.mean(x, axis)
    return np.ones_like(y)*100

b = sns.tsplot(data=df, value="lifetimes", unit="run", time="indices",
               estimator=np.median,
               # err_kws={"alpha": 0.4},
               # err_style="unit_traces",
               ci=95,
               legend=False,
               ax=ax[0]
               )

# ax.set_title("Time before rolling")
# ax.set_ylabel(r"$\mathregular{t}$")
ax[0].set_ylabel(r"Time before rolling")
# ax.set_xlabel("$\mathregular{T}$")
ax[0].set_xlabel("Descendant")
# ax.set_xlabel("$\mathregular{T}$\nold ancestors$\\longrightarrow$new descendants")
# ax.set_xticks([])
ax[0].set_xlim([0, 120])
ax[0].set_ylim([-0.25, 10.25])


time = np.array(discovery_time_begin+discovery_time_end)
group = np.array([0]*len(discovery_time_begin) + [1]*len(discovery_time_end))
df = pd.DataFrame({'time': time, 'Descendant': group})

print np.median(discovery_time_begin)

g = sns.barplot(x='Descendant', y="time", data=df, capsize=0.1, errwidth=2.5, ax=ax[1], estimator=np.median, ci=95)
ax[1].set_ylabel(r"Time before rolling")
ax[1].set_xticklabels(['First to roll', 'Run champion'])
ax[1].set_ylim(ax[0].get_ylim())


ax[0].text(ax[0].get_xlim()[1]*0.95, ax[0].get_ylim()[1], "A", ha="right", va="top", fontsize=30, fontname="Arial")
ax[1].text(ax[1].get_xlim()[1]*0.95, ax[1].get_ylim()[1], "B", ha="right", va="top", fontsize=30, fontname="Arial")


sns.despine()
plt.tight_layout()
plt.savefig("plots/Journal_Discovery_Time.tiff", bbox_inches='tight', format='tiff', dpi=4500/12)


