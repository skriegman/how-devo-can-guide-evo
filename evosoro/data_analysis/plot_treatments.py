import numpy as np
from itertools import product
import ast
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import BlendedGenericTransform
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_palette("Set2", 8)
# print sns.color_palette()
# sns.axes_style()

start_vol = 2
final_vol = 7
midlife = (start_vol + final_vol) / 2
frequency = 8/(2*np.pi)
amplitude = 0.5
start_phase = 7  # np.pi/2.
final_phase = 2  # -np.pi/2.
growth_onset = 3
growth_offset = 7
color = (0.70196080207824707, 0.70196080207824707, 0.70196080207824707)  # sns.color_palette()[7]

index = np.arange(11)
evo = np.array([start_vol]*11)

high_res_index = np.linspace(0, 11, 10000)
global_actuation = amplitude * np.sin(2 * np.pi * frequency * high_res_index)
low_res_evo = np.array([start_vol]*10000)
high_res_evo = low_res_evo + amplitude*np.sin(2*np.pi*frequency*high_res_index+np.pi/2.)
low_res_evo_phase = np.array([start_phase]*10000)

phase_shift = np.linspace(np.pi/2., -np.pi/2., 10000)
devo_actuation = amplitude*np.sin(2*np.pi*frequency*high_res_index+phase_shift)
low_res_devo = np.linspace(start_vol, final_vol, 10000)
high_res_evodevo = low_res_devo + devo_actuation
low_res_devo_phase = np.linspace(start_phase, final_phase, 10000)


#########################

evodevo_velocity = []
evodevo_velocity_idx = []
evo_velocity = []
evo_velocity_idx = []

for RUN in [7, 6]:

    if RUN not in [6, 8, 16, 17, 18]:
        RUN_DIR = "/home/sam/Archive/skriegma/Ball_Discovery/"
        first_gen_stats = pd.read_table(RUN_DIR + 'run_{}/allIndividualsData/Gen_0000.txt'.format(RUN),
                                        delimiter="\t\t", lineterminator='\n', engine='python')

        lifetime = first_gen_stats['lifetime'].values.tolist()
        idx = first_gen_stats['id'].values.tolist()
        lifetime = [x - 0.5 for _, x in sorted(zip(idx, lifetime), reverse=True)]

        t = 0
        found = False
        while not found:
            if lifetime[t] < 10:
                found = True
                peak = t
            t += 1

    DIM = 0.01
    BODY_LENGTH = 4.0
    STEP_SIZE = 49

    df = pd.read_csv('results/Traces_Devo.csv', low_memory=False)
    df = df[df['run'] == RUN-1]

    velocities = []

    for trace in df['trace'][::-1]:
        this_trace = np.array(ast.literal_eval(trace)) / (DIM * BODY_LENGTH)
        velocity = np.zeros(len(this_trace))

        for step in range(STEP_SIZE, len(this_trace), STEP_SIZE):
            velocity[step] = np.linalg.norm(this_trace[step] - this_trace[step-1])

        velocities.append(velocity[::STEP_SIZE])

    # SCALE_CHANGE = 7.5/np.max(velocities[peak])
    SCALE_CHANGE = 5/(STEP_SIZE/1000.0)

    evodevo_velocity += [velocities[-1]*SCALE_CHANGE]
    evodevo_velocity_idx += [np.linspace(0, 11, len(velocities[-1]))]
    # evodevo_velocity += [velocities[peak] * SCALE_CHANGE]
    # evodevo_velocity_idx += [np.linspace(0, 11, len(velocities[peak]))]

    df1 = pd.read_csv('results/Traces_Evo.csv', low_memory=False)
    velocities = []

    for trace in df1['trace']:
        this_trace = np.array(ast.literal_eval(trace)) / (DIM * BODY_LENGTH)
        velocity = np.zeros(len(this_trace))

        for step in range(STEP_SIZE, len(this_trace), STEP_SIZE):
            velocity[step] = np.linalg.norm(this_trace[step] - this_trace[step-1])

        velocities.append(velocity[::STEP_SIZE])

    evo_velocity += [velocities[RUN-1]*SCALE_CHANGE]
    
    evo_velocity_idx += [np.linspace(0, 11, len(velocities[RUN-1]))]

    V_MARKER = 1.7  # 1 body length per second
    print V_MARKER

#############################


fig, axes = plt.subplots(5, 2, figsize=(8, 14))

# Evo
for v in range(len(evo_velocity)):
    axes[4, 0].plot(evo_velocity_idx[v], evo_velocity[v], zorder=1, color=sns.color_palette()[0])
axes[3, 0].plot(high_res_index, high_res_evo, zorder=1, color=sns.color_palette()[0])
axes[2, 0].plot(high_res_index, low_res_evo, zorder=1, color=sns.color_palette()[0])
axes[1, 0].plot(high_res_index, low_res_evo_phase, zorder=1, color=sns.color_palette()[0])
axes[0, 0].plot(high_res_index, 5+global_actuation, zorder=1, color=sns.color_palette()[0])


axes[4, 0].set_xticklabels([])
axes[4, 0].set_yticks([V_MARKER])
axes[4, 0].set_yticklabels([1], fontsize=20)
axes[4, 0].set_ylabel("Velocity", fontsize=18)
axes[4, 0].axhline(V_MARKER, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)


axes[3, 0].axhline(start_vol, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[3, 0].set_xticklabels([])
axes[3, 0].set_yticks([start_vol])
axes[3, 0].set_yticklabels(['$\ell_k$'], fontsize=20)
axes[3, 0].set_ylabel("Current Length", fontsize=18)


axes[2, 0].axhline(start_vol, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[2, 0].set_xticklabels([])
axes[2, 0].set_yticks([start_vol])
axes[2, 0].set_yticklabels(['$\uparrow$\n$\ell_k$\n$\downarrow$'], fontsize=20)
axes[2, 0].set_ylabel("Resting Length", fontsize=18)


axes[1, 0].axhline(start_phase, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[1, 0].set_xticklabels([])
axes[1, 0].set_yticks([start_phase])
axes[1, 0].set_yticklabels(['$\uparrow$\n$\mathregular{\phi}_k$\n$\downarrow$'], fontsize=20)
axes[1, 0].set_ylabel("Phase Offset", fontsize=18)


axes[0, 0].axhline(5, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[0, 0].set_xticklabels([])
axes[0, 0].set_yticks([5])
axes[0, 0].set_yticklabels(['0'], fontsize=20)
axes[0, 0].set_ylabel("Global\nTemperature", fontsize=18)


# Evo-Devo
for v in range(len(evodevo_velocity)):
    axes[4, 1].plot(evodevo_velocity_idx[v], evodevo_velocity[v], zorder=1, color=sns.color_palette()[1])
axes[3, 1].plot(high_res_index, high_res_evodevo, zorder=1, color=sns.color_palette()[1])
axes[2, 1].plot(high_res_index, low_res_devo, zorder=1, color=sns.color_palette()[1])
axes[1, 1].plot(high_res_index, low_res_devo_phase, zorder=1, color=sns.color_palette()[1])
axes[0, 1].plot(high_res_index, 5+global_actuation, zorder=1, color=sns.color_palette()[1])


axes[4, 1].set_xticklabels([])
axes[4, 1].set_yticklabels([])
axes[4, 1].axhline(V_MARKER, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)


axes[3, 1].axhline(start_vol, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[3, 1].axhline(final_vol, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[3, 1].set_yticks([start_vol, final_vol])
axes[3, 1].set_yticklabels(['$\ell_k$', '$\ell^*_k$'], fontsize=20)
axes[3, 1].set_xticks([])


axes[2, 1].axhline(start_vol, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[2, 1].axhline(final_vol, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[2, 1].set_xticklabels([])
axes[2, 1].set_yticks([start_vol, final_vol])
axes[2, 1].set_yticklabels(['$\uparrow$\n$\ell_k$\n$\downarrow$', '$\uparrow$\n$\ell_k^*$\n$\downarrow$'], fontsize=20)


axes[1, 1].axhline(start_phase, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[1, 1].axhline(final_phase, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[1, 1].set_xticklabels([])
axes[1, 1].set_yticks([start_phase, final_phase])
axes[1, 1].set_yticklabels(['$\uparrow$\n$\mathregular{\phi}_k$\n$\downarrow$',
                            '$\uparrow$\n$\mathregular{\phi}^*_k$\n$\downarrow$'], fontsize=20)


axes[0, 1].axhline(5, color=color, linestyle='dashed', linewidth=1.75, alpha=0.8, zorder=0)
axes[0, 1].set_xticklabels([])
axes[0, 1].set_yticks([5])
axes[0, 1].set_yticklabels(['0'], fontsize=20)


axes[0, 0].set_title("Evo", fontsize=18)
axes[0, 1].set_title("Evo-Devo", fontsize=18)
axes[4, 0].set_xlabel("Lifetime", fontsize=18)

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
for a in range(5):
    for b in range(2):
        axes[a, b].set_ylim([0, 10])
        axes[a, b].text(1.0, -0.8, '$\mathregular{t}$',
                        transform=BlendedGenericTransform(axes[a, b].transAxes, axes[a, b].transData),
                        va='center', fontsize=20)
        axes[a, b].set_aspect('equal')
        axes[a, b].text(1, 8, labels[b*5+a], fontsize=30, fontname="Arial")
        # print axes[a, b].get_xlim()

axes[4, 0].text(7.5, 6, "run 6", fontsize=15, ha="left", va="center")
axes[4, 0].text(7.5, 2.5, "run 7$\\ast$", fontsize=15, ha="left", va="center")
axes[4, 1].text(7.5, 9, "run 7", fontsize=15, ha="left", va="center")
axes[4, 1].text(7.5, 3.75, "run 6$\\ast$", fontsize=15, ha="left", va="center")

sns.despine()
plt.tight_layout()
plt.savefig("plots/Journal_Treatments.tiff", format='tiff', dpi=4500*0.6/8)
