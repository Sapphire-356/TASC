# Author: Weinan He
# Mail: Sapphire9877@gmail.com
# ----------------------------------------------


import copy
import numpy as np
from math import log
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from nltk import data
data.path.append(r"data/nltk_data")
from nltk.corpus import wordnet
# nltk.download('wordnet')
import os
from sklearn.metrics import roc_auc_score
import random
from collections import Counter
from tqdm import tqdm
from typing import Literal, Tuple


from train_sapphire_CLIPLoRABase \
import BasicInfo, CLIPLoRABaseTrainer, CLIPLoRABase, TextEncoder

from sapphire.datasets import *
from sapphire.models import *
from sapphire.train import *
from sapphire.test import *
from sapphire.utils import *
from sapphire.templates import get_templates
from sapphire.models.backbone import CLIP_MODELS

import clip



class TASCBaseTrainer(CLIPLoRABaseTrainer):

    def __init__(self, conf, writer, dir_dict, logger):
        super().__init__(conf, writer, dir_dict, logger)
        self.thr_curr = None


    def build_model(self):
        self.model = TASCBase(self.info)


    def writer_known_score_hist(self, iter_num, known_scores, gt_labels):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        known_score_dict = {
            'k': known_scores[gt_labels < self.info.num_classes],
            'u': known_scores[gt_labels >= self.info.num_classes],
        }
        try: 
            histogram_writer(self.info.writer, iter_num, known_score_dict, tag=f'known_score', binwidth=0.010)
        except: 
            self.info.logger.info("writer_known_score_hist() failed!")


    def get_multiple_unknown_scores(self, ind, all_results, gt_labels):
        try:
            for key in all_results.keys():
                if "unknown_scores" not in key:
                    continue
                metric_name = key.split("_")[-1]
                unknown_scores = torch.cat(all_results[key])
                auroc = roc_auc_score(1 - np.array(gt_labels < self.info.num_classes)*1, unknown_scores)
                self.info.writer.add_scalar(f'multiple_unknown_scores/auroc_{metric_name}', auroc, global_step=self.iter_num)
                self.metrics[ind]['Unknown_binary'][metric_name] = 100*auroc
        except:
            self.info.logger.info("calculate multiple_unknown_scores, failed!")


    def set_transforms(self, stage, domain, aug_type: Literal["strong", "weak"] = None):
        '''
        Set the transforms used in dataloader.
        '''
        flag = stage + '-' + domain
        if aug_type is not None:
            flag = flag + '-' + aug_type
        aug_list = self.info.conf.dataset.augmentation
        data_transforms = [get_data_transforms(aug, self.info.conf.model.name) for aug in aug_list]
        val_data_transforms = [get_data_transforms('clip-none', self.info.conf.model.name)]
        transforms_dict = {
            'train-source': data_transforms,
            'train-target': data_transforms,
            'val-source': val_data_transforms,
            'val-target': val_data_transforms,
        }
        return transforms_dict[flag]



class TASCBase(CLIPLoRABase):

    @torch.no_grad()
    def init_memory(self, loader, feat_only=True):

        self.info.logger.info(f"init memory")
        loader = self.loaders['test'][0]
        loader_iter = iter(loader)

        num_samples = len(loader.dataset)
        gt_labels = torch.zeros((num_samples,), dtype=torch.int64).to(self.device)
        features = torch.zeros((num_samples, self.feature_dim)).to(self.device)
        if not feat_only:
            probs = torch.zeros((num_samples, self.info.num_classes)).to(self.device)
            clu_probs = torch.zeros((num_samples, self.num_clusters)).to(self.device)

        self.before_testing()
        for _ in range(len(loader)):
            batched_input = next(loader_iter)
            results = self._predict(batched_input, feat_only=feat_only)
            indexs = results['indexs']
            gt_labels[indexs] = batched_input['gt_label'].to(self.device)
            features[indexs] = results['feat']
            if not feat_only:
                probs[indexs] = results['prob']
                clu_probs[indexs] = results['prob_cluster']

        if feat_only:
            return (features, gt_labels)
        else:
            return (features, gt_labels, probs, clu_probs)


    @torch.no_grad()
    def _predict(self, batched_input, feat_only=False):
        
        results = {}
        images = batched_input['img'].to(self.device)
        indexs = batched_input['data_info']['image_ind'].to(self.device)
        images = images if images.ndim == 4 else images.unsqueeze(0)
        feat = self.backbone(images)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        if feat_only:
            results['feat'] = feat
            results['indexs'] = indexs
            return results

        if self.test_flag == False:
            self.init_text_embeddings()
            self.test_flag = True

        logits = feat @ self.text_features.T / self.ce_prob_temp
        logits_cluster = feat @ self.text_features_target.T / self.ce_prob_temp
        prob = F.softmax(logits, dim=1)
        prob_cluster = F.softmax(logits_cluster, dim=1)

        results['gt_labels'] = batched_input['gt_label']
        results['feat'] = feat
        results['prob'] = prob
        results['prob_cluster'] = prob_cluster

        return results


    def before_forward(self, iter_num):
        self.backbone.train()
        self.text_encoder.train()
        self.iter_num = iter_num
    

    def text_forward(self, classnames):

        text_indexs = torch.randint(self.n_templates, (len(classnames),))
        text_templates = self.templates[text_indexs]

        prompts = [template.format(name) for name, template in \
                   zip(classnames, text_templates)]

        tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) \
                                       for p in prompts]).to(self.device)
        embeddings = self.clip_model_token_embedding(tokenized_prompts)

        text_features = self.text_encoder(embeddings, tokenized_prompts)
        text_features = F.normalize(text_features)

        return text_features


    def before_testing(self):
        self.backbone.eval()
        self.text_encoder.eval()
        self.test_flag = False


    def after_testing(self,):
        del self.text_features
        del self.text_features_target
        torch.cuda.empty_cache()


    @torch.no_grad()
    def get_all_nouns_features(self):

        self.nouns_np = get_nouns_from_wordnet()
        num_nouns = self.nouns_np.size
        path_extracted = "data/WordNet/nouns_feat_rm_redundancy_ensemble.pth"

        if os.path.exists(path_extracted):
            self.info.logger.info(f"load extracted features from '{path_extracted}'")
            all_nouns_features = torch.load(path_extracted)
            
            return all_nouns_features.to(self.device)

        self.info.logger.info(f"'{path_extracted}' does not exist, "
                              f"starting text_encoder forward()")
        # Note: you can also replace 'ensemble' with 'vanilla' or 'classname' 
        #   to save time
        templates = get_templates(self.info.dataset_name, 'ensemble')
        all_nouns_features = torch.zeros((num_nouns, self.feature_dim))

        self.before_testing()
        for i in tqdm(range(num_nouns)):
            noun = self.nouns_np[i]
            prompts = [template.format(noun) for template in templates]
            tokenized_prompts = \
                torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            embeddings = self.clip_model_token_embedding(tokenized_prompts)
            text_features = self.text_encoder(embeddings, tokenized_prompts)
            text_features = F.normalize(text_features)
            text_features = text_features.mean(dim=0, keepdim=True)
            text_features = F.normalize(text_features)
            all_nouns_features[i] = text_features.squeeze().detach().cpu()
            if i % 50 == 0: 
                torch.cuda.empty_cache()

        torch.save(all_nouns_features, path_extracted)

        return all_nouns_features.to(self.device)


def get_entropy(features, anchors, temp=0.01, normalized=True):

    probs = F.softmax(features @ anchors.T / temp, dim=1)
    entropy = -(probs * (probs + 1e-5).log()).sum(dim=1)
    if normalized:
        return entropy / log(probs.size(1))
    else:
        return entropy


def get_prototypes(image_features, classifier):
    assert image_features.ndim == 2
    assert classifier.ndim == 3
    logits = image_features @ classifier.mT
    clu_pred = torch.max(logits, dim=-1)[1]  # K_s x N
    unique_labels = torch.unique(clu_pred, dim=-1)  # K_s x K
    cluster_mask_bool = (clu_pred.unsqueeze(2) == unique_labels.unsqueeze(1))
    cluster_mask = F.normalize(1.0*cluster_mask_bool, p=1, dim=1)
    prototypes = F.normalize(cluster_mask.mT @ image_features, p=2, dim=-1)
    return prototypes


def get_nouns_from_wordnet():
    ## extract nouns from WordNet
    nouns = list(wordnet.all_synsets(pos='n'))
    nouns = [noun.name().split('.')[0] for noun in nouns]
    nouns = list(Counter([noun for noun in nouns]))  # use Counter to eliminate redundancy
    nouns = [noun.replace("_", " ") for noun in nouns]  # nouns in which "_" has been replaced by " "

    return np.char.array(nouns)


# copy from https://github.com/pytorch/pytorch/issues/35674
def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

