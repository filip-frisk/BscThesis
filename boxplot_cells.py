from accuracy_dict import all_cells as data
import matplotlib as mpl
mpl.rcParams['boxplot.medianprops.color'] = 'black'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# This code generates and saves Figure 11 and 12 of the thesis 

def set_ax_and_colors(ax,bp,lim):
	ax.grid(axis = "y")
	ax.set_xticklabels([])
	ax.set_xticks([])
	ax.set_ylim(lim)
	for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)

metrics = ['precision', 'recall', 'f1-score']
metrics_names = ['Precision', 'Recall', 'F1-score']
classes = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial','epithelial']
class_names = ['Inflammatory', 'Lymphocyte', 'Fibroblast and endothelial','Epithelial']
filter_names = ["No filter","Macenko","Reinhard","Khan","Ensemble naive","Ensemble PSA"]

#print([[[len(data[metric][filter][c]) for c in data[metric][filter].keys()] for filter in data[metric].keys()] for metric in data.keys()])

colors = ['#E5C0BB', '#81C2F3','#89F381', '#EB7FF3', '#F3B381', '#F3EC81']
ylims = [[0, 1],[0, 1],[0, 1],[0, 1]]

fig, ax = plt.subplots(3, 4,figsize=(17, 8))
for i, (metric,m_name, ax1) in enumerate(zip(metrics,metrics_names,ax)):
	for j, (c, c_name, ax2) in enumerate(zip(classes,class_names,ax1)):
		ax2.set_xlabel(c_name)
		ax2.set_ylabel(m_name)
		bp = ax2.boxplot([data[filter][c][metric] for filter in data.keys()], patch_artist = True)
		set_ax_and_colors(ax2,bp,ylims[i])

handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors,filter_names)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles))
plt.savefig('Boxplot_metrics_on_class_per_model.png',dpi=300,bbox_inches = 'tight',pad_inches=0.2)
plt.show()

colors = ['#E36B6B', '#E2E269','#69E2E2','#6969E2']
fig, ax = plt.subplots(3, 6,figsize=(17, 8))

for i, (metric,m_name, ax1) in enumerate(zip(metrics,metrics_names,ax)):
	for j, (filter, f_name, ax2) in enumerate(zip(data.keys(),filter_names,ax1)):
		ax2.set_xlabel(f_name)
		if j == 0: ax2.set_ylabel(m_name)
		bp = ax2.boxplot([data[filter][c][metric] for c in classes], patch_artist = True)
		set_ax_and_colors(ax2,bp,ylims[i])

handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors,class_names)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles))
plt.savefig('Boxplot_metrics_on_model_per_class.png',dpi=300,bbox_inches = 'tight',pad_inches=0.2)
plt.show()