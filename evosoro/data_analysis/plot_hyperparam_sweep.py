import numpy as np
import pandas as pd
import seaborn as sns
from fractions import Fraction
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
sns.set_palette("Set2", 8)


ROWS, COLS = 2, 4
f, axes = plt.subplots(ROWS, COLS, figsize=(20, 10))

NAMES = ["1", "2", "4", "8", "16", "24", "32", "48"]

for ax, NAME in enumerate(NAMES):
    print "NAME ", NAME
    row, col = np.unravel_index(ax, (ROWS, COLS))

    df = pd.read_csv('results/best_results_co_opt_{}.csv'.format(NAME), low_memory=False)
    df = df.fillna(0)
    df.ix[df.fitness <= 0, 'fitness'] = 0

    df['fitness'] *= 0.25  # body lengths

    orig_len = len(df)
    print len(df[df.duplicated(subset=["condition", "gen", "run"])]) == len(df[df.duplicated(subset=["id", "condition", "gen", "run", "fitness"])])
    print len(df[df.duplicated(subset=["condition", "gen", "run"])]), " duplicates"
    df = df.drop_duplicates(subset=["condition", "gen", "run"], keep='last')

    a = sns.tsplot(data=df, value="fitness", condition="condition", unit="run", time="gen", ax=axes[row, col],
                   ci=95,
                   err_style=[
                       # "unit_traces",
                       "ci_band"
                   ],
                   estimator=np.median,
                   legend=False
                   )

    frac = Fraction(float(NAME)/48.).limit_denominator()
    axes[row, col].annotate("$\lambda={}$".format(str(frac)), xy=(400, 68), fontsize=30)

    axes[row, col].grid(color='lightgray', lw=1)
    axes[row, col].set_ylabel("")
    axes[row, col].set_xlabel("")
    axes[row, col].set_ylim([0, 80])
    axes[row, col].set_xlim([-50, 10000+50])
    axes[row, col].set_xticks(range(1000, 10000, 1000))
    axes[row, col].set_yticks(range(10, 80, 10))
    if row == ROWS-1:
        axes[row, col].xaxis.set_ticklabels([1000, "", 3000, "", 5000, "", 7000, "", 9000, ""], fontsize=17)
    else:
        axes[row, col].xaxis.set_ticklabels([])

    if col == 0:
        axes[row, col].yaxis.set_ticklabels([10, 20, 30, 40, 50, 60, 70, ""], fontsize=17)
    else:
        axes[row, col].yaxis.set_ticklabels([])


axes[0, 0].set_ylabel("o", fontsize=30)
axes[1, 1].set_xlabel("                               Generation", fontsize=30)
axes[0, 0].legend([mpatches.Patch(color=sns.color_palette()[i]) for i in range(2)], ['Evo', 'Evo-Devo'], loc=[0.0, 0.5], fontsize=30)

f.text(0.0155, 0.75, 'Body lengths traveled', rotation=90, fontsize=30,
       bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1, 'edgecolor': 'white'})

# f.legend([mpatches.Patch(color=sns.color_palette()[i]) for i in range(2)],
#          ['Evo', 'Evo-Devo'], loc=[0.3735, 0.945], ncol=2, fontsize=26)

plt.tight_layout()
# plt.axis('equal')
f.subplots_adjust(wspace=0, hspace=0)
plt.savefig("plots/Journal_Hyper_Param_Sweep_Median.tiff", bbox_inches='tight', format='tiff', dpi=4500/20)

