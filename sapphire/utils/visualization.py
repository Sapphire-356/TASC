import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score, auc


__all__ = ["histogram_writer", ]



def histogram_writer(writer, iter_num, data_dict, tag='tag', kde=False, binwidth=None, binrange=None, fill=True,):
    '''
        draw a histogram of each data in data_dict, and display fig in tensorboard 'histogram'.

        args:
            iter_num: global_step
            data_dict: data with this format:
                {
                    <label>: <data>(tensor,numpy array or list)
                    ...
                } 
            kde: kernel density estimation
    '''
    plt.clf()
    for name, data in data_dict.items():
        temp = sns.histplot(data, label=name, kde=kde, binwidth=binwidth, binrange=binrange, fill=fill)
    plt.legend() 
    fig = temp.get_figure()
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.clf()
    writer.add_image('histogram/{}'.format(tag), image, global_step=iter_num, dataformats='HWC')
    plt.close()
