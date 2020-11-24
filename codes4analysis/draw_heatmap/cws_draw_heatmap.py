import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import string
# sphinx_gallery_thumbnail_number = 2
import argparse


# row_name_list = []
# col_name_list = []
# mat = []
# for ind, line in enumerate(sys.stdin):
#     line = line.strip()
#     if ind == 0:
#         row_name_list = line.split("\t")
#     elif ind == 1:
#         col_name_list = line.split("\t")
#     else:
#         val_list = [round(string.atof(w),4) for w in line.split("\t")]
#         mat.append(val_list)
    
# print('mat:',mat)
# mat_array = np.array(mat)
# print('mat_array: ',mat_array)

'''
vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.1111", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
'''

def heatmap(attrs_name,data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.tick_params(labelsize=25)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels) 

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False,labelsize=30, pad=15)
    legend = ax.legend()
    legend.remove()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, va="center",
             rotation_mode="anchor")

    # plt.setp(ax.get_yticklabels(), rotation=-90, ha="right", visible=True, rotation_mode="anchor")
    
    if attrs_name == 'MF-et' : # or attrs_name == 'R-eDen'
        plt.setp(ax.get_yticklabels(), ha="right", visible=True, rotation_mode="anchor")
    else:
        plt.setp(ax.get_yticklabels(), ha="right", visible=False, rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

if __name__ == '__main__':
    # result_dir = 'results/'
    result_dir = '../evaluation_results/cws-evaluation/'
    result_files = os.listdir(result_dir)
    dict_aided_diag_heatmap = {}
    new_corpus_name_dic = {'ctb':'ctb','ckip':'ckip','cityu':'cityu',
                            'ncc':'ncc','sxu':'sxu','pku':'pku'}
    for file in result_files:
        corpus_name = new_corpus_name_dic[file.split('_')[0]]
        print('corpus_name: ',corpus_name)
        fn_path = result_dir+file
        print('fn_path: ',fn_path)
        fread = open(fn_path, 'r')
        file_string = fread.read()
        for block in file_string.split("# "):
            if block.find("aided-diagnosis heatmap") != -1:
                for ind, line in enumerate(block.split("\n")):
                    info_list = line.strip().split("\t")
                    if 'heatmap' in info_list:
                        continue
                    if len(info_list) <=1:
                        continue
                    att_name, val_list = info_list
                    # dict_aided_diag_heatmap[att_name][corpus_name] = val_list

                    if att_name not in dict_aided_diag_heatmap:
                        dict_aided_diag_heatmap[att_name] = {}
                    if corpus_name not in dict_aided_diag_heatmap[att_name]:
                        dict_aided_diag_heatmap[att_name][corpus_name] = val_list
                    else:
                        print('22222',info_list)

    print('dict_aided_diag_heatmap: ',dict_aided_diag_heatmap)

    xsticks = ['XS','S','L','XL']
    ysticks = ['ctb','ckip','cityu','ncc','sxu','pku']
    attrs_names = ['MF-et', 'MF-tt', 'F-ent', 'F-tok', 'R-eLen', 'R-sLen', 'R-oov']
    for attrs_name in attrs_names:
        attr_corpus_4values_list = []
        for corpus_name in ysticks:
            value = dict_aided_diag_heatmap[attrs_name][corpus_name]
            value_float = [float(v) for v in value.split()]
            attr_corpus_4values_list.append(value_float)
        # heatmap(attr_corpus_4values_list, xsticks, ysticks, ax=None,
        #     cbar_kw={}, cbarlabel="", **kwargs)

        #print mat
        out_fig = 'cws-heatmap/'+attrs_name+'.png'
        fig, ax = plt.subplots()
        im, cbar = heatmap(attrs_name,np.array(attr_corpus_4values_list), xsticks, ysticks, ax=ax, vmin=-0.005, vmax=0.005,
                           cmap='RdBu')  # cmap ="PiYG"
        if attrs_name not in ['R-oov']: #'F-tok',
            cbar.remove()
        fig.tight_layout()

        plt.savefig(out_fig,bbox_inches = 'tight',
                    pad_inches = 0)




# parser = argparse.ArgumentParser(description='Draw Bar')
# parser.add_argument('--png', default='out.png', help='output image file')

# args = parser.parse_args()

# out_fig = unicode(args.png)

# #print mat
# fig, ax = plt.subplots()
# im, cbar = heatmap(mat_array, row_name_list, col_name_list, ax=ax, vmin=-0.03, vmax=0.03,
#                    cmap="PiYG")
# #                   cmap="PiYG", cbarlabel="Fine-grained Evaluation")

# '''
# im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
#                    cmap="Blues", cbarlabel="harvest [t/year]")
# '''
# #texts = annotate_heatmap(im, valfmt="{x:.1f} t")
# #cbar.remove()
# fig.tight_layout()

# plt.savefig(out_fig,bbox_inches = 'tight',
#             pad_inches = 0)
