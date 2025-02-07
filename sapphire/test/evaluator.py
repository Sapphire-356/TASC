

import torch

from sklearn.cluster import KMeans
import numpy as np
import copy

import torch.nn.functional as F

from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def run_kmeans(L2_feat, ncentroids, init_centroids=None, gpu=False, min_points_per_centroid=1):
    dim = L2_feat.shape[1]
    import faiss
    kmeans = faiss.Kmeans(d=dim, k=ncentroids, gpu=gpu, niter=20, verbose=False, \
                        nredo=5, min_points_per_centroid=min_points_per_centroid, spherical=True)
    if torch.is_tensor(L2_feat):
        L2_feat = L2_feat.cpu().detach().numpy()
    kmeans.train(L2_feat, init_centroids=init_centroids)
    _, pred_centroid = kmeans.index.search(L2_feat, 1)
    pred_centroid = np.squeeze(pred_centroid)
    return pred_centroid, kmeans.centroids

    # kmeans = KMeans(n_clusters=ncentroids, n_init=5, max_iter=20)
    # if torch.is_tensor(L2_feat):
    #     L2_feat = L2_feat.cpu().detach().numpy()
    # kmeans.fit(L2_feat)
    # pred_centroid = kmeans.predict(L2_feat)
    # centroids = kmeans.cluster_centers_
    # return pred_centroid, centroids


def get_curve_online(known: np.ndarray, novel: np.ndarray, stypes = ['Bas']):
    known = known.copy()
    novel = novel.copy()
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1: np.ndarray, x2: np.ndarray, stypes = ['Bas'], verbose=False):
    x1 = x1.copy()
    x2 = x2.copy()
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(-np.trapz(1.-fpr, tpr))  # This might be incorrect. In np.trapz(y, x=None), 1.-fpr is considered as y by default. (by xxx)
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')
    
    return results


def compute_oscr(x1, x2, pred_in, labels_in):
    x1 = x1.copy()
    x2 = x2.copy()
    m_x1 = np.zeros(len(x1))
    m_x1[pred_in == labels_in] = 1  # This is the sole relevant information from these two variables that can be used to calculate OSCR

    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)  # pred labels of known samples, that is, pred labels masked by gt unknown
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)  # gt labels of all samples, 1 represents unknown

    predict = np.concatenate((x1, x2), axis=0)  # known pred score (known as pos)
    n = len(predict)

    # Cutoffs are of prediction values
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort()

    # sort the elements in k_target and u_target according to idx
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]
    thr = predict[idx]

    for k in range(n-1):
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # TruePositive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)  # sorted according to FPR

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR, ROC, thr


def cluster_acc(y_true, y_pred, verbose=False):
    """
    modified by https:
    copy from https://github.com/DonkeyShot21/UNO
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    from scipy.optimize import linear_sum_assignment
    
    def compute_best_mapping(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        y_true_unique = np.unique(y_true)
        y_pred_unique = np.unique(y_pred)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1  # mapping: pred -> true
        # In UniDA, y_true may miss some classes
        # Here, we set cost=-inf to the missing classes
        for i in range(D):
            if not np.any(y_true_unique == i):
                w[:, i] += -int(1e7)
            if not np.any(y_pred_unique == i):
                w[i, :] += -int(1e7)
        return np.transpose(np.asarray(linear_sum_assignment(-w))), w
    
    mapping, w = compute_best_mapping(y_true, y_pred)  # mapping: pred -> true
    y_pred_unique = np.unique(y_pred)
    mapping_ = np.zeros((y_pred_unique.size, 2), dtype=np.int64)
    j = 0
    for i in y_pred_unique:
        mapping_[j] = mapping[i]
        j += 1

    acc = sum([w[i, j] for i, j in mapping_]) * 1.0 / y_pred.size
    if verbose: 
        print(f"cluster acc: {acc*100:0.2f}")
    return mapping_


def cluster_metrics(clu_pred, target_features=None, gt_labels=None, verbose=False):
    metrics_dict = {}
    clu_pred = clu_pred.cpu() if clu_pred.device != 'cpu' else clu_pred
    if target_features is not None:
        target_features = target_features.cpu() if target_features.device != 'cpu' else target_features
        metrics_dict['silhouette score'] = silhouette_score(target_features, clu_pred)
    if gt_labels is not None:
        gt_labels = gt_labels.cpu() if gt_labels.device != 'cpu' else gt_labels
        metrics_dict['AMI'] = adjusted_mutual_info_score(gt_labels, clu_pred) 
        metrics_dict['NMI'] = normalized_mutual_info_score(gt_labels, clu_pred)
        metrics_dict['ARI'] = adjusted_rand_score(gt_labels, clu_pred)
    print_str = ''
    for key in metrics_dict.keys():
        print_str += f"{key}: {metrics_dict[key]*100:0.2f}, "
    if verbose: 
        print(print_str, end="")
    
    return metrics_dict


def get_silhouette_score(labels, all_dists, num_clusters=1e5):
    '''
    Created by https://

    Parallelize the calculation by linear algebra to accelerate.

    Space Complexity Analysis:
        Notations: N, the number of samples. K, the number of unique labels in `labels`.
        
        all_dists: O(N^2)
        cluster_mask: O(K*N)

    '''
    unique_labels = torch.unique(labels)
    num_samples = len(labels)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")
    # Although number of unique labels may not equal to `num_clusters`, each sample
    # always has a cluster label.
    assert len(unique_labels) <= num_clusters
    assert all_dists.size(0) == all_dists.size(1)
    assert all_dists.size(0) == labels.size(0)

    # `cluster_mask` is crucial for Parallelization
    cluster_mask_bool = (labels.unsqueeze(1) == unique_labels)
    cluster_mask = F.normalize(1.0*cluster_mask_bool, p=1, dim=0) 

    scores = torch.zeros_like(labels).float()
    ## calculate the `b`
    sample2clu_dists = all_dists @ cluster_mask  # "num_samples" x "number of unique labels"
    sample2clu_dists_mask_self = sample2clu_dists + 1e5*cluster_mask_bool  # "num_samples" x "number of unique labels"
    sample_min_other_dists = torch.min(sample2clu_dists_mask_self, dim=1)[0]
    b = sample_min_other_dists  # `b` follows the standard definition in Silhouette Score
    ## calculate the `a` for the clusters with more than 1 elements
    sample2clu_dists_mask_other = sample2clu_dists[torch.nonzero(cluster_mask_bool, as_tuple=True)]
    num_elements = cluster_mask_bool.sum(dim=0)
    num_elements_sample_wise = (cluster_mask_bool * num_elements.unsqueeze(0)).sum(dim=1)
    mt1_mask = num_elements_sample_wise > 1  # "mt1" means more than 1
    # In line 3 below, we substract 1 from the number to exclude self distance
    sample2clu_dists_mask_other[mt1_mask] = sample2clu_dists_mask_other[mt1_mask] \
                                            * num_elements_sample_wise[mt1_mask] \
                                            / (num_elements_sample_wise[mt1_mask] - 1)
    a = sample2clu_dists_mask_other
    ## calculate the Silhouette Score
    # Set score=0 for all the samples assigned in the clusters which num_elements=0.
    # Here, do nothing because of the zero initialization of `scores`.
    scores_temp = (b - a) / (torch.maximum(a, b))
    scores[mt1_mask] = scores_temp[mt1_mask]
    silhouette = torch.mean(scores).item()

    return silhouette