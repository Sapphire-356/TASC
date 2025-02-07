# Author: Weinan He
# Mail: Sapphire9877@gmail.com
# ----------------------------------------------


import copy
import numpy as np
from math import log
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import os.path as osp
from sklearn.metrics import roc_auc_score


from train_sapphire_CLIPLoRABase import BasicInfo
from train_sapphire_TASCBase import TASCBaseTrainer, TASCBase
from train_sapphire_TASCBase import get_prototypes, get_entropy, unravel_indices

from sapphire.datasets import *
from sapphire.models import *
from sapphire.train import *
from sapphire.test import *
from sapphire.utils import *



class TASCTrainer(TASCBaseTrainer):

    def build_model(self):
        self.model = TASC(self.info)


    def test(self, iter_num):
        loaders = self.loaders['test']
        self.model.before_testing()

        for loader_ind in range(len(loaders)):
            self.info.logger.info(f"Testing, loader:{loader_ind}")
            all_results = {}
            first_predict = True
            loader_iter = iter(loaders[loader_ind])
            for _ in range(len(loaders[loader_ind])):
                batched_inputs = next(loader_iter)
                results = self.model.predict(batched_inputs)
                if first_predict:
                    keys = results.keys()
                    for key in keys:
                        all_results[key] = []
                    first_predict = False 
                for key in keys:
                    all_results[key].append(results[key].cpu())
            
            known_scores = torch.cat(all_results["known_scores"])
            gt_labels = torch.cat(all_results["gt_labels"])
            predict_labels = torch.cat(all_results["predict_labels_without_unknown"])
            predict_labels_without_unknown = copy.deepcopy(predict_labels)
            known_scores = (known_scores - known_scores.min()) / \
                           (known_scores.max() - known_scores.min())

            ## Determine the threshold to detect unknown samples
            weights_init = np.array([self.model.estimated_shared, \
                                    self.model.num_clusters - \
                                    self.model.estimated_shared])
            weights_init = weights_init / weights_init.sum()

            thr = gmm_threshold(known_scores, 
                                weights_init, 
                                self.model.fixed_weights,
                                thr_curr=self.thr_curr,
                                )

            ## Evaluate
            self.thr_curr = thr
            known_mask = known_scores > thr
            predict_labels[~known_mask] = self.info.num_classes
            
            bs = len(all_results["gt_labels"][0])
            self.evaluate(loader_ind, 
                          all_results["gt_labels"], 
                          all_results["logits"], 
                          all_results["feat"], 
                          torch.split(known_scores, bs), 
                          torch.split(predict_labels, bs),
                          torch.split(predict_labels_without_unknown, bs))
            
            self.print_key_metric(iter_num, loader_ind)
            self.get_multiple_unknown_scores(loader_ind, all_results, gt_labels)
            self.save_metric(iter_num)
            self.writer_known_score_hist(iter_num, known_scores, gt_labels)

        self.model.after_testing()

    
def gmm_threshold(known_scores, 
                  weights_init,
                  fixed_weights,
                  thr_curr = None,
                  reset_weights = True,
                  momentum = 0.5
                  ):
    _scores = known_scores.unsqueeze(1)
    _scores = torch.nan_to_num(_scores, nan=0.5, posinf=1, neginf=0)

    gm = CustomGaussianMixture(n_components=2,
                                random_state=0,
                                weights_init=weights_init,
                                means_init=np.array([[0.75], [0.25]]),
                                covariance_type="spherical",
                                fixed_weights=fixed_weights,
                                verbose=False,
                                verbose_interval=1,
                                n_init=3).fit(_scores)
    mi, ma = gm.means_.min(), gm.means_.max()
    print(f"gm.means_: {gm.means_.squeeze()}, gm.weights_: {gm.weights_}, "
          f"gm.covariance_: {gm.covariances_}")

    # reset the weights to balance the OS* and UNK (see our Appendix)
    if reset_weights:
        gm.weights_ = np.array([0.5, 0.5])
    
    # calculate the threshold
    rang = np.array([i for i in np.arange(mi, ma, 0.001)]).reshape(-1, 1)
    temp = gm.predict(rang)
    thr_new = np.array(temp).mean()*(ma - mi) + mi
    thr_curr = thr_new if thr_curr is None else thr_curr
    thr = (1 - momentum)*thr_new + momentum*thr_curr

    print(f"thr: {thr:0.2f}, thr_curr: {thr_curr:0.2f}, "
          f"thr_new: {thr_new:0.2f}")

    return thr



class TASC(TASCBase):

    def before_training(self):
        super().before_training()

        self.lambda_dict = {
            'loss_s_ce': self.loss_s_ce,
            'loss_t_im_ent': self.loss_t_im_ent,
            'loss_t_im_div': self.loss_t_im_div,
        }

        ## Discovery the classname
        self.source_classnames = self.classnames_split['source']
        self.discovered_classnames, self.num_clusters, self.estimated_shared =\
                                                    self.discrete_optimization()


    def discrete_optimization(self):

        self.info.logger.info(f"Starting Discrete Optimization")
        
        # hyper-parameters
        num_clusters_dict = self.TASC.num_clusters_dict
        K_0 = num_clusters_dict[self.info.dataset_name]
        n_inner = self.TASC.n_inner
        K_s = 100 if self.info.dataset_name == 'domainnet' else self.TASC.K_s

        print(f"initial num_clusters = {K_0}")

        ## extract target features
        # gt labels are only used to evaluate the performance of TASC
        self.info.logger.info(f"extract target features")
        target_features, target_gt_labels = \
                                    self.init_memory(self.loaders['test'][0])

        ## get features of source classnames
        self.info.logger.info(f"get features of source classnames")
        source_text_features = \
                        self.embed_classnames(self.classnames_split['source'],
                                            need_token=False,)

        ## get features of all nouns in WordNet
        all_nouns_features = self.get_all_nouns_features()

        ## Select better nouns by Greedy Search
        print(f"inner: {n_inner}, K_s: {K_s}")
        if self.info.dataset_name == 'domainnet':  # just to keep the code robust
            self.classnames_split['target'] = ['none']
            self.classnames_split['target_private'] = ['none']
        
        ## Execute the greedy search
        (self.nouns_iter, self.S_iter, self.r_iter, 
        self.num_clusters, self.estimated_shared) = \
            greedy_search(self.info,
                          target_features, 
                          all_nouns_features, 
                          self.nouns_np, 
                          K_0,
                          gt_labels=target_gt_labels,
                          n_inner=n_inner,
                          K_s=K_s,
                          W_src=source_text_features,
                          W_src_classnames=\
                                np.char.array(self.classnames_split['source']),
                          metric_temp=self.metric_temp,
                          ent_temp=self.ent_temp,
                          ent_thr=self.ent_thr,
                          K_beam=self.TASC.K_beam,
                          lambda_dict=self.lambda_dict,
                          n_keep_status=self.n_keep_status)
        nouns_matched, nouns_features_matched = \
            self.nouns_iter[self.r_iter.cpu().numpy()], self.S_iter[self.r_iter]

        ## Evaluation
        # evalute matched nouns
        print(f"Matched performance:")
        logits = target_features @ nouns_features_matched.T
        clu_pred = torch.max(logits, dim=1)[1]
        cluster_metrics(clu_pred, 
                        target_features=None, 
                        gt_labels=target_gt_labels, 
                        verbose=True)
        print("")
        self.info.logger.info(f"greedy searching, done")
        self.to(self.device)
        # evaluate gt classnames
        self.info.logger.info(f"evaluate gt classnames")
        self.info.logger.info(f"get features of ground truth classnames")
        gt_nouns_features = \
                        self.embed_classnames(self.classnames_split['target'],
                                              need_token=False)
        pred_from_gt_classname = \
                    torch.max(target_features @ gt_nouns_features.T, dim=1)[1]
        print(f"gt classnames: ", end='')
        cluster_metrics(pred_from_gt_classname, 
                        target_features=None, 
                        gt_labels=target_gt_labels, 
                        verbose=True,)
        cluster_acc(target_gt_labels.cpu().numpy(), 
                    pred_from_gt_classname.cpu().numpy(), 
                    verbose=True,)

        ## target classname
        save_path = osp.join(self.info.log_dir, "target_classnames.txt")
        with open(save_path, "w") as file:
            file.write("\n".join(list(nouns_matched)))
        self.info.logger.info(f"finished TASC algorithm")
        torch.cuda.empty_cache()

        return nouns_matched, self.num_clusters, self.estimated_shared


    def forward(self, batched_inputs):
        '''
        return: loss_dict
        '''
        loss_dict = {}

        source_images = batched_inputs[0]['aug'][0].to(self.device)
        target_images = batched_inputs[1]['img'].to(self.device)

        # Text forward
        text_features = self.text_forward(self.source_classnames)
        text_features_target = self.text_forward(self.discovered_classnames)

        # Source forward
        source_feat = self.backbone(source_images)
        source_feat = source_feat / source_feat.norm(dim=-1, keepdim=True)
        source_logits = (source_feat @ text_features.T) / self.ce_prob_temp

        # Source Cross Entropy Loss
        source_labels = batched_inputs[0]['gt_label'].to(self.device)
        loss_s_ce = F.cross_entropy(source_logits, 
                                    source_labels, 
                                    label_smoothing=0.1)
        loss_dict['loss_s_ce'] = loss_s_ce

        # Target forward
        target_feat = self.backbone(target_images)
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        cluster_logits = (target_feat @ text_features_target.T)
        cluster_logits /= self.ce_prob_temp
        loss_t_im_ent, loss_t_im_div = shot_im_loss(cluster_logits)
        loss_dict["loss_t_im_ent"] = loss_t_im_ent
        loss_dict["loss_t_im_div"] = loss_t_im_div

        for _, key in enumerate(loss_dict):
            loss_dict[key] *= self.lambda_dict[key]

        return loss_dict


    def init_text_embeddings(self,):
        self.before_testing()

        self.text_features = \
            self.embed_classnames(self.classnames_split['source'], 
                                  need_token=False)
        self.text_features_target = \
            self.embed_classnames(self.discovered_classnames,
                                  need_token=False)
        self.n_templates = len(self.templates)

        # relations between source classnames and target classnames
        self.info.logger.info(f"extract relations between source classnames "
                              f"and target classnames")
        t2s_logits = self.text_features_target @ \
                                    self.text_features.T / self.text_temp
        s2t_logits = self.text_features @ \
                                    self.text_features_target.T / self.text_temp
        self.t2s_prob = F.softmax(t2s_logits, dim=1)
        self.s2t_prob = F.softmax(s2t_logits, dim=1)
        self.t2s_ent = (-self.t2s_prob*self.t2s_prob.log()).sum(dim=1).squeeze()
        self.t2s_ent /= log(self.t2s_prob.size(1))
        self.s2t_ent = (-self.s2t_prob*self.s2t_prob.log()).sum(dim=1).squeeze()
        self.s2t_ent /= log(self.s2t_prob.size(1))

        torch.cuda.empty_cache()


    def get_unknown_scores(self, logits, logits_cluster, s2t_ent, t2s_ent):
        unknown_scores_dict = {}
        unknown_scores_dict['MS-s'] = -torch.max(logits, dim=1)[0]
        unknown_scores_dict['MS-t'] = torch.max(logits_cluster, dim=1)[0]
        unknown_scores_dict['MS-s-w/ent'] = \
                        -torch.max(logits*(1 - s2t_ent).unsqueeze(0), dim=1)[0]
        unknown_scores_dict['MS-t-w/ent'] = \
                        torch.max(logits_cluster*t2s_ent.unsqueeze(0), dim=1)[0]
        unknown_scores_dict['UniMS'] = \
                    torch.max(logits_cluster*t2s_ent.unsqueeze(0), dim=1)[0] - \
                    torch.max(logits*(1 - s2t_ent).unsqueeze(0), dim=1)[0]
        
        return list(unknown_scores_dict.keys()), unknown_scores_dict


    @torch.no_grad()
    def predict(self, batched_input, feat_only=False):
        
        results = {}
        images = batched_input['img'].to(self.device)
        indexs = batched_input['data_info']['image_ind'].to(self.device)
        images = images if images.ndim == 4 else images.unsqueeze(0)
        feat = self.backbone(images)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        if feat_only == True:
            results['feat'] = feat
            results['indexs'] = indexs
            return results
        
        if self.test_flag == False:
            self.init_text_embeddings()
            self.test_flag = True

        logits = feat @ self.text_features.T / self.ce_prob_temp
        logits_cluster = feat @ self.text_features_target.T / self.ce_prob_temp

        _, predict_labels = torch.max(logits, -1)
        _, predict_labels_cluster = torch.max(logits_cluster, -1)
        predict_labels_without_unknown = copy.deepcopy(predict_labels)

        _, unknown_score_dict = self.get_unknown_scores(logits, 
                                                        logits_cluster, 
                                                        self.s2t_ent, 
                                                        self.t2s_ent)

        results['gt_labels'] = batched_input['gt_label']
        results['feat'] = feat
        results['predict_labels'] = predict_labels
        results['predict_labels_cluster'] = predict_labels_cluster
        results['logits'] = logits
        results['logits_cluster'] = logits_cluster
        results['predict_labels_without_unknown'] = predict_labels_without_unknown
        results['known_scores'] = -unknown_score_dict['UniMS']
        results['indexs'] = indexs

        for key in unknown_score_dict.keys():
            results[f'unknown_scores_{key}'] = unknown_score_dict[key]

        return results



def greedy_search(info: BasicInfo,
                  image_features,
                  nouns_features,
                  nouns,
                  K_0,
                  *,
                  gt_labels=None,
                  n_inner=3,
                  K_s=1000,
                  W_src=None,
                  W_src_classnames=None,
                  metric_temp=0.01,
                  ent_temp=0.01,
                  ent_thr=None,
                  K_beam=10,
                  lambda_dict=None,
                  n_keep_status=5,
                  ):

    num_samples, feat_dim = image_features.size()
    device = image_features.device
    assert len(W_src) == len(W_src_classnames)
    assert len(W_src) <= K_0

    num_nouns = nouns_features.size(0)
    num_src_names = W_src.size(0)

    ######   Greedy Search   ######
    S_iter = torch.empty(K_beam, K_0, feat_dim).to(device)
    r_iter = torch.empty(K_beam, K_0, dtype=bool).to(device)
    nouns_iter = np.empty((K_beam, K_0), dtype=nouns.dtype)

    rand_indices = torch.randperm(num_samples)
    rand_indices = rand_indices[:min(5000, num_samples)]
    image_features_p = image_features[rand_indices]
    gt_labels_p = gt_labels[rand_indices]

    ## Initializations
    for b in range(K_beam):
        rand_index = torch.randperm(num_nouns)[:K_0 - num_src_names].to(device)
        S_iter[b] = torch.cat([W_src, nouns_features[rand_index]])
        r_iter[b] = torch.ones(K_0) > 0
        nouns_iter[b] = np.concatenate([W_src_classnames, \
                                        nouns[rand_index.cpu().numpy()]])
        # If r_i = 1, $s′_i$ is retained; if ri = 0, s′_i is discarded.

    ###  Inner Iteration  ###
    for inner in range(n_inner):

        if inner < n_keep_status:
            keep_status = True
        else:
            keep_status = False

        ## Go through all the candidates once
        for i in range(K_0):

            # Speed up by sampling
            rand_indices = torch.randperm(num_samples)
            rand_indices = rand_indices[:min(5000, num_samples)]
            image_features_p = image_features[rand_indices]
            gt_labels_p = gt_labels[rand_indices]

            if i < num_src_names:
                results_features = W_src[i:i+1].unsqueeze(0).repeat(K_beam, 1, 1)  # K_beam x n_candidates x D
                results_nouns = np.expand_dims(W_src_classnames[i:i+1], axis=0).repeat(K_beam, 0)  # K_beam x n_candidates
            else:
                results_features = []
                results_nouns = [] 
                for b in range(K_beam):
                    ind_results = torch.randperm(num_nouns).to(device)[:K_s]
                    _results_features = nouns_features[ind_results]
                    _results_nouns = nouns[ind_results.cpu().numpy()]
                    _results_features = torch.cat((S_iter[b, i:i+1, :], _results_features))  # K_beam x n_candidates x D
                    _results_nouns = np.concatenate((nouns_iter[b, i:i+1], _results_nouns))  # K_beam x n_candidates
                    results_features.append(_results_features)
                    results_nouns.append(_results_nouns)
                results_features = torch.stack(results_features)
                results_nouns = np.stack(results_nouns)

            n_cand = results_nouns.shape[1]

            ## Evaluate the performance of all candidates.
            S_iter_temp = S_iter.unsqueeze(1).repeat(1, n_cand, 1, 1)  # K_beam x n_candidates x K_0 x D
            S_iter_temp[:, :, i, :] = results_features
            r_iter[:, i] = True | r_iter[:, i]
            M_beam_k = []
            for S_iter_b, r_iter_b in zip(S_iter_temp, r_iter):
                M_k = Metric(S_iter_b, r_iter_b, image_features_p, lambda_dict, ce_prob_temp=metric_temp)
                M_beam_k.append(M_k)
            M_beam_k = torch.stack(M_beam_k)  # K_beam x n_candidates
            
            ## Evaluate the performance when s′_i is discarded.
            r_iter[:, i] = False & r_iter[:, i]  # -> discarded
            M_beam_dis = []
            for S_iter_b, r_iter_b in zip(S_iter, r_iter):
                M_dis = Metric(S_iter_b, r_iter_b, image_features_p, lambda_dict, ce_prob_temp=metric_temp)
                M_beam_dis.append(M_dis)
            M_beam_dis = torch.stack(M_beam_dis)  # (K_beam, )

            if ent_thr is not None and i < num_src_names:
                r_iter[:, i] = True | r_iter[:, i]
                for b in range(K_beam):
                    prototypes = get_prototypes(image_features_p, S_iter[b][r_iter[b]].unsqueeze(0)).squeeze(0)
                    ent_best = get_entropy(S_iter[b, i:i+1, :], prototypes, temp=ent_temp).squeeze()
                    if ent_best < ent_thr:
                        M_beam_dis[b] -= 1e5

            # top K_beam
            if not keep_status:
                M_beam = torch.cat((M_beam_k, M_beam_dis), dim=1)
                M_beam_best, k_beam_best = torch.topk(M_beam.flatten(), k=K_beam)
                k_beam_best = unravel_indices(k_beam_best, M_beam.shape)
            else:
                M_beam_best, k_beam_best = torch.topk(M_beam_k, k=1, dim=1)
                M_beam_best = M_beam_best.squeeze(1)
                k_beam_best = torch.stack([torch.arange(K_beam).to(device), k_beam_best.squeeze(1)], dim=1)
            
            # top K_beam without M_beam_dis
            _, k_beam_best_wo_dis = torch.topk(M_beam_k, k=1, dim=1)

            ## Update
            new_S_iter = torch.empty_like(S_iter)
            new_r_iter = torch.empty_like(r_iter)
            new_nouns_iter = np.empty_like(nouns_iter)
            for b in range(K_beam):
                ind_beam, ind_cand = k_beam_best[b]
                ind_beam, ind_cand = int(ind_beam), int(ind_cand)
                new_S_iter[b] = S_iter[ind_beam]
                new_nouns_iter[b] = nouns_iter[ind_beam]
                new_r_iter[b] = r_iter[ind_beam]
                if ind_cand < n_cand:  # not discarded
                    new_S_iter[b, i] = results_features[ind_beam, ind_cand]
                    new_nouns_iter[b, i] = results_nouns[ind_beam, ind_cand]
                    new_r_iter[b, i] = True
                else:  # if discarded, choose the best one in M_beam_k[b]
                    new_S_iter[b, i] = results_features[ind_beam, int(k_beam_best_wo_dis[ind_beam])]
                    new_nouns_iter[b, i] = results_nouns[ind_beam, int(k_beam_best_wo_dis[ind_beam])]
                    new_r_iter[b, i] = False

            S_iter = new_S_iter
            r_iter = new_r_iter
            nouns_iter = new_nouns_iter
            K = (1*r_iter).sum(dim=1)

            k_beam_best_str = ''
            for b in range(K_beam):
                k_beam_best_str += f"({int(k_beam_best[b, 0])}, {int(k_beam_best[b, 1])}), "
            print(f"cluster {i:>3}, n: {results_nouns.shape}, k_best: " + k_beam_best_str + f"K: {int(K)}, M: {M_beam_best.squeeze():0.6f}, ent_best: {ent_best<ent_thr}")
            torch.cuda.empty_cache()


        ## Evaluation
        logits = image_features_p @ S_iter[0][r_iter[0]].T
        clu_pred = torch.max(logits, dim=1)[1]
        cluster_metrics(clu_pred, target_features=None, gt_labels=gt_labels_p, verbose=True)

        print(f"inner: {inner}, K: {int(K[0])}\n")
        
    S_iter = S_iter[0]
    r_iter = r_iter[0]
    nouns_iter = nouns_iter[0]
    K = int((1*r_iter).sum())

    estimated_shared = torch.nonzero(r_iter[:num_src_names]).size(0)
    print(f"finally, K: {K}, shared-fp: {(1*~r_iter[:info.shared]).sum()}/{info.shared}, private-fn: {torch.nonzero(r_iter[info.shared:info.num_classes]).size(0)}/{info.source_private}")
    print(f"greedy search, done!\n")

    return nouns_iter, S_iter, r_iter, K, estimated_shared


def Metric(S_iter, r_iter, image_features, lambda_dict, ce_prob_temp=0.01):

    loss_dict = {}

    S_iter = S_iter.unsqueeze(0) if S_iter.ndim != 3 else S_iter
    assert S_iter.ndim == 3
    assert r_iter.ndim == 1

    logits = (image_features @ S_iter[:, r_iter, :].mT) / ce_prob_temp  # K_s x N x K

    loss_t_im_ent, loss_t_im_div = shot_im_loss(logits, normalized=True)
    loss_dict["loss_t_im_ent"] = loss_t_im_ent
    loss_dict["loss_t_im_div"] = loss_t_im_div

    score = 0
    for _, key in enumerate(loss_dict):
        loss_dict[key] *= lambda_dict[key]
        score += loss_dict[key]

    return -score


def shot_im_loss(logits, normalized=False):

    probs = F.softmax(logits, dim=-1)
    mprobs = probs.mean(dim=-2, keepdim=True)
    multi = 1 / log(probs.size(-1)) if normalized else 1
    entropy = -(probs * (probs + 1e-5).log()).sum(dim=-1) * multi
    mentropy = -(mprobs * (mprobs + 1e-5).log()).sum(dim=-1) * multi

    entropy = entropy.mean(dim=-1)
    mentropy = mentropy.squeeze(-1)

    return entropy, -mentropy

