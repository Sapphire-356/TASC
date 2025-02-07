
import json
import copy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import beta as beta_f
import torch
import torch.nn.functional as F



def save_as_json(obj, fpath):
    '''Save an object as json.'''
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": ")) 


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):  # p(k)*p(l|k) == p(y)*p(x|y)
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        self.score_history = []
        self.weight_0 = []
        self.weight_1 = []
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            neg_log_likelihood = np.sum([self.score_samples(i) for i in x])
            self.score_history.append(neg_log_likelihood)
            self.weight_0.append(self.weight[0])
            self.weight_1.append(self.weight[1])
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l

    def look_lookup(self, x, loss_max, loss_min, testing=False):
        if testing:
            x_i = x
        else:
            x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self, title, save_dir, save_signal=False):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='y=0')
        plt.plot(x, self.weighted_likelihood(x, 1), label='y=1')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()
        if save_signal:
            plt.title(title)
            plt.savefig(save_dir, dpi=300)
        plt.close()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

    def calculate_criteria(self):
        self.K = ( self.weight[0] * beta_f(self.alphas[1], self.betas[1])) / ( self.weight[1] * beta_f(self.alphas[0], self.betas[0]))
        self.criteria = ((np.log(self.K)) - (self.betas[1] - self.betas[0])) / ( (self.alphas[1]-self.alphas[0]) - (self.betas[1]-self.betas[0]) )
        print(self.K, self.alphas[1]-self.alphas[0], beta_f(2,3))
        return self.criteria


class BetaMixture1DCutoffScaling(BetaMixture1D):
    '''
    Using 'cut_off' to make the fitting process easier and enhance the performance.
    '''

    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5],
                 cut_off=[0.0, 1.0],
                 scalar: float = None,
                 ):
        super().__init__(max_iters, alphas_init, betas_init, weights_init)
        self.cut_off = cut_off
        if scalar == None:
            self.scalar = 1 / (self.cut_off[1] - self.cut_off[0])
        else:
            self.scalar = scalar
        print(f"BetaMixture1D with CutoffScaling: cut_off {self.cut_off}, scalar {self.scalar}")


    def encode(self, x: np.array, cut_off: bool = False):
        x = copy.deepcopy(x)
        if cut_off == True:
            x = x[x <= self.cut_off[1]]
            x = x[x > self.cut_off[0]]
        elif cut_off == False:
            x[x > self.cut_off[1]] = self.cut_off[1]
            x[x < self.cut_off[0]] = self.cut_off[0]
        x = (x - self.cut_off[0]) * self.scalar

        return x
    

    def decode(self, x):
        x = copy.deepcopy(x)
        x /= self.scalar 
        x += self.cut_off[0]
        return x
    

    def plot(self, title, save_dir, save_signal=False):
        x = np.linspace(0, 1, 100)
        x_true = self.decode(x)
        plt.plot(x_true, self.weighted_likelihood(x, 0), label='y=0')
        plt.plot(x_true, self.weighted_likelihood(x, 1), label='y=1')
        plt.plot(x_true, self.probability(x), lw=2, label='mixture')
        plt.legend()
        if save_signal:
            plt.title(title)
            plt.savefig(save_dir, dpi=300)
        plt.close()


def js_div(p: torch.Tensor, q: torch.Tensor):
    assert p.ndim <= 2
    assert q.ndim <= 2
    p = p if p.ndim==2 else p.unsqueeze(0) 
    q = q if q.ndim==2 else q.unsqueeze(0) 
    m = 0.5*(p + q)
    kl_1 = F.kl_div(m.log(), p, reduction='none').sum(dim=1, keepdim=True)
    kl_2 = F.kl_div(m.log(), q, reduction='none').sum(dim=1, keepdim=True)
    return (kl_1 + kl_2)*0.5