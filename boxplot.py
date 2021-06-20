from accuracy_dict import all
import matplotlib as mpl
mpl.rcParams['boxplot.medianprops.color'] = 'black'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# This code generates and saves Figure 9 and 10 of the thesis 

metric = 'accuracy'
#print([[len(all[metric][filter]) for filter in all[metric].keys()] for metric in all.keys()])
colors = ['#E5C0BB', '#81C2F3','#89F381', '#EB7FF3', '#F3B381', '#F3EC81']
filter_names = ["No filter","Macenko","Reinhard","Khan","Ensemble naive","Ensemble PSA"]

def set_ax_and_colors(ax,bp):
	ax.grid(axis = "y")
	ax.set_xticklabels([])
	ax.set_xticks([])

	for patch, color in zip(bp['boxes'], colors):
		patch.set_facecolor(color)



fig, ax = plt.subplots(figsize=(9, 6))
metric = 'accuracy'
ax.set_ylabel("Accuracy", fontsize=15)
ax.set_ylim([0.48, 0.65])
bp = plt.boxplot([all[metric][filter] for filter in all[metric].keys()], patch_artist = True)
set_ax_and_colors(ax,bp)
handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors,filter_names)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(filter_names)//2)
plt.savefig('Boxplot_accuracy.png',dpi=300)

fig, ax = plt.subplots(2, 3,figsize=(17, 8))
((ax1, ax2, ax3), (ax4, ax5, ax6)) = ax
axes = [ax1,ax2,ax3,ax4,ax5,ax6]
average = ""
metrics = ['precision_ma', 'recall_ma', 'f1-score_ma', 'precision_wma','recall_wma','f1-score_wma']
metrics_names = ['Macro average precision', 'Macro average recall', 'Macro average f1-score', 'Weighted macro average precision','Weighted macro average recall','Weighted macro average f1-score']

for metric, ax, name in zip(metrics,axes, metrics_names):
	ax.set_ylabel(name)
	ax.set_ylim([0.35, 0.65])
	bp = ax.boxplot([all[metric][filter] for filter in all[metric].keys()], patch_artist = True)
	set_ax_and_colors(ax,bp)


handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors,filter_names)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=len(filter_names))
plt.savefig('Boxplot_metrics.png',dpi=300,bbox_inches = 'tight',pad_inches=0.2)