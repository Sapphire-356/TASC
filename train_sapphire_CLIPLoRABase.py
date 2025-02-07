# Author: Weinan He
# Mail: Sapphire9877@gmail.com
# ----------------------------------------------


import numpy as np
from math import log
from abc import ABC, abstractmethod
import copy
import time
import os
import random
import easydict
import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import recall_score, average_precision_score, \
    normalized_mutual_info_score, roc_auc_score

from sapphire.models.backbone import CLIP_MODELS

from sapphire.datasets import *
from sapphire.models import *
from sapphire.train import *
from sapphire.test import *
from sapphire.utils import *
from sapphire.templates import get_templates

import clip

CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']


class BasicInfo:

    def __init__(self, 
                 conf: easydict.EasyDict, 
                 writer: SummaryWriter, 
                 dir_dict: dict, 
                 logger: logging.Logger):
        
        self.conf = conf
        self.writer = writer
        self.dir_dict = dir_dict
        self.data_dir = dir_dict['data']
        self.project_dir = dir_dict['project']
        self.log_dir = dir_dict['log']
        self.logger = logger

        # dataset config
        self.dataset_name = conf.dataset.name  
        self.task = conf.dataset.task
        self.shared = conf.dataset.shared
        self.source_private = conf.dataset.source_private
        self.target_private = conf.dataset.target_private
        self.unknown = self.shared + self.source_private
        self.num_classes = self.unknown



class CLIPLoRABaseTrainer(ABC):

    def __init__(self, conf, writer, dir_dict, logger):

        self.info = BasicInfo(conf, writer, dir_dict, logger)
        self.num_classes = self.info.num_classes
        self.unknown = self.info.unknown
        self.metrics = []
        self.method_args = conf.method_args
        self.amp = (hasattr(conf, 'amp') and conf.amp == True)
        if self.amp == True:
            self.scaler = GradScaler()


    def build_model(self):
        self.model = CLIPLoRABase(self.info)
    
    
    def build_optimizer(self):
        cfg = self.info.conf.optimizer 
        self.optimizers = []

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        # fix all params without lora params
        params_lora = []
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                params_lora.append(param)

        optim_global = optim.SGD(params_lora, 
                                 lr=cfg.backbone_lr,
                                 momentum=cfg.momentum, 
                                 weight_decay=cfg.weight_decay, 
                                 nesterov=True)
        self.optimizers.append(optim_global)


    def build_lr_schedule(self):
        cfg = self.info.conf.lr_schedule
        self.lr_schedules = []
        for i in range(len(self.optimizers)):
            lr_schedule = build_lr_scheduler(self.optimizers[i], 
                                             cfg.type, 
                                             cfg.warmup_iter, 
                                             cfg.max_iter, 
                                             warmup_type=cfg.warmup_type, 
                                             warmup_lr=cfg.warmup_min_lr,
                                             lr_cfg = cfg.lr_cfg)
            self.lr_schedules.append(lr_schedule)
    
    
    def build_data_loaders(self):
        self.info.logger.debug("start build data_loaders")
        cfg = self.info.conf.dataset
        self.num_workers = cfg.num_workers
        balanced = cfg.balanced
        batchsize = cfg.batchsize
        val_batchsize = cfg.val_batchsize

        task_source, task_target = get_task(cfg.name, cfg.task)
        class_split = make_class_split(cfg.name, 
                                       cfg.shared, 
                                       cfg.source_private, 
                                       cfg.target_private)

        self.datasets = {
            'train': [],
            'test': [],
        }

        self.info.logger.debug("require_source == True")
        source_dataset = UniversalDataset(self.info.data_dir, 
                                          cfg.name, 
                                          task_source, 
                                          source=True,
                                          data_transforms=self.set_transforms('train', 'source'), 
                                          class_split=class_split)
        self.datasets['train'].append(source_dataset)

        self.info.logger.debug("require_target == True")
        target_dataset = UniversalDataset(self.info.data_dir, 
                                          cfg.name, 
                                          task_target, 
                                          source=False,
                                          data_transforms=self.set_transforms('train', 'target'), 
                                          class_split=class_split)
        self.datasets['train'].append(target_dataset)

        if cfg.val_source == True:
            self.info.logger.info("val_source == True, source domain -> self.metrics[1]")
            val_source_dataset = UniversalDataset(self.info.data_dir, 
                                                  cfg.name, 
                                                  task_source, 
                                                  source=True,
                                                  data_transforms=self.set_transforms('val', 'source'), 
                                                  class_split=class_split) 
            
        val_target_dataset = UniversalDataset(self.info.data_dir, 
                                              cfg.name, 
                                              task_target, 
                                              source=False,
                                              data_transforms=self.set_transforms('val', 'target'), 
                                              class_split=class_split)
         
        self.datasets['test'].append(val_target_dataset)                              

        self.loaders = {
            'train': [],
            'test': [],
        }
        if self.model.require_source == True:
            batchsize_source = round(batchsize/2.0)
            if balanced:
                freq = Counter(source_dataset.labels)
                class_weight = {x: 1.0 / freq[x] for x in freq}
                source_weights = [class_weight[x] for x in source_dataset.labels]
                sampler = WeightedRandomSampler(source_weights,
                                                len(source_dataset.labels))
                self.info.logger.debug("use balanced loader")
                source_loader = DATALoader(source_dataset,
                                           batch_size=batchsize_source,
                                           sampler=sampler,
                                           drop_last=True,
                                           num_workers=self.num_workers)
                self.loaders['train'].append(source_loader)
            else:
                source_loader = DATALoader(source_dataset,
                                           batch_size=batchsize_source,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=self.num_workers)
                self.loaders['train'].append(source_loader)

        if self.model.require_target == True:
            batchsize_target = round(batchsize/2.0)
            target_loader = DATALoader(target_dataset,
                                       batch_size=batchsize_target,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=self.num_workers)
            self.loaders['train'].append(target_loader)

        val_target_loader = DATALoader(val_target_dataset,
                                       batch_size=val_batchsize,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=self.num_workers)
        self.loaders['test'].append(val_target_loader)

        if cfg.val_source == True:
            # source -> self.metrics[1]
            val_source_loader = DATALoader(val_source_dataset,
                                           batch_size=val_batchsize,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=self.num_workers)
            self.loaders['test'].append(val_source_loader)


    def train(self):
        cfg = self.info.conf.train
        control = self.info.conf.control
        loaders = self.loaders['train']
        loaders_iter = []
        self.iter_num = 0
        self.model.iter_num = 0

        # init
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.loaders = self.loaders
        self.model.datasets = self.datasets
        self.model.before_training()
        for iter_num in range(control.max_iter):
            self.iter_num = iter_num
            self.model.iter_num = iter_num
            
            # update
            if iter_num % control.interval == 0:
                self.model.before_interval(iter_num)
            self.model.custom_interval()

            # get batched_inputs
            try: self.time_data_in = time.clock()
            except: self.time_data_in = time.perf_counter()
            batched_inputs = []
            for i in range(len(loaders)):
                if iter_num == 0:
                    loaders_iter.append(iter(loaders[i]))
                elif iter_num % len(loaders[i]) == 0:
                    loaders_iter[i] = iter(loaders[i])
                batched_inputs.append(next(loaders_iter[i]))
            try: self.time_data_out = time.clock()
            except: self.time_data_out = time.perf_counter()

            # forward
            self.model.before_forward(iter_num)

            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()
            
            # loss
            try: self.time_in = time.clock()
            except: self.time_in = time.perf_counter()
            if self.amp == True:
                with autocast():
                    loss_dict = self.model(batched_inputs)
            else:
                loss_dict = self.model(batched_inputs)
            try: self.time_out = time.clock()
            except: self.time_out = time.perf_counter()
            loss = 0
            for _, key in enumerate(loss_dict):
                loss += loss_dict[key]
            loss_dict['loss'] = loss

            # backward
            if self.amp == True:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            for i in range(len(self.optimizers)):
                if self.amp == True:
                    self.scaler.step(self.optimizers[i])
                else:
                    self.optimizers[i].step()
            if self.amp == True:
                self.scaler.update()
            for i in range(len(self.lr_schedules)):
                self.lr_schedules[i].step()  # remove iter_num
            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()

            self.model.after_backward(iter_num)
            self.writer_log_(iter_num, loss_dict)
            
            # log
            if iter_num % control.log_interval == 0:
                self.info.logger.info(f"start recording log, iter_num: {iter_num}")
                self.log_(iter_num, loss_dict)
                self.writer_log_(iter_num, loss_dict)
            
            # update, testing
            if iter_num % control.interval == 0:
                self.info.logger.info(f"start testing, iter_num: {iter_num}")
                self.model.after_interval(iter_num)
                self.test(iter_num)
                self.info.logger.info(f"start writer_interval, iter_num: {iter_num}")
                self.writer_interval(iter_num)
                self.info.logger.info(f"testing finished, iter_num: {iter_num}")
            
        self.test(iter_num)
        # save
        self.model.after_training()


    @abstractmethod
    def test(self, iter_num):  
        return
    

    def set_transforms(self, stage, domain):
        '''
        Set the transforms used in dataloader.
        '''
        flag = stage + '-' + domain
        transforms_dict = {
            'train-source': get_data_transforms('clip-randomcrop', self.info.conf.model.name),
            'train-target': get_data_transforms('clip-randomcrop', self.info.conf.model.name),
            'val-source': get_data_transforms('clip-none', self.info.conf.model.name),
            'val-target': get_data_transforms('clip-none', self.info.conf.model.name),
        }
        return transforms_dict[flag]
    

    def evaluate(self, loader_ind, gt_labels, logits, feat, known_scores, \
                 predict_labels, predict_labels_without_unknown):
        '''
        To avoid ambiguity, the overall accuracy (multi-class accuracy) will be 
        denoted as OA, and the average classes accuracy (multi-class recall) 
        will be denoted as OS according to the notation used in OSR.
        
        '''
        result =   {'Closed-set':{  # calculate on target-shared samples
                        'OA': None,
                        'Recall': None,
                        'class-wise': None,
                    },
                    'Open-set':{  
                        'OA': None,  # overall accuracy of (num_classes + 1) classes
                        'OS*': None,  # average classes accuracy of target-shared samples
                        'class-wise': None,
                        'UNK': None,  # unk recall
                        'OS': None,  # average classes accuracy of (num_classes + 1) classes
                        'NMI': None,  # NMI of target-private samples
                    },
                    'Unknown_binary':{  # when known as pos
                        'TP': None,
                        'FN': None,
                        'FP': None,
                        'TN': None,
                        'AUROC': None,
                        'AUPR': None,  # AUPR when unknown as pos
                        'AUPR-neg': None,  # note that AUPR is sensitive to pos\neg ratio
                    },
                    'UniDA':{
                        'H-score': None,  # harmonic mean of OS* and UNK
                        'H3-score': None,  # harmonic mean of OS*, UNK and NMI
                        'UCR': None,  # UCR from paper uniood
                    }
                  }
        
        self.OSCR_ROC = None
        gt_labels = torch.cat(gt_labels)
        logits = torch.cat(logits)
        predict_labels = torch.cat(predict_labels)

        label_set = set(gt_labels.tolist())
        private_label_set = label_set - set(range(self.num_classes))
        n_target_private = len(private_label_set)

        np_grds = copy.copy(gt_labels.numpy())
        np_prds = copy.copy(predict_labels.numpy())
        target_private_indexs = [True if lab in private_label_set else False \
                                 for lab in gt_labels.tolist()]
        target_shared_indexs = [False if id else True for id in target_private_indexs]
        np_grds[target_private_indexs] = self.num_classes

        # metric
        if n_target_private == 0:
            if len(predict_labels_without_unknown) > 0:
                self.info.logger.info('n_target_private == 0, Closed-set setting.')
                prds_without_unknown = torch.cat(predict_labels_without_unknown)
                np_prds_without_ood = copy.copy(prds_without_unknown.numpy()) 
                recall_avg_auc = recall_score(np_grds, np_prds_without_ood, labels=np.unique(np_grds), average=None)
                OA = np.mean(np_grds==np_prds_without_ood)
                result['Closed-set']['OA'] = float(100.*OA)
                result['Closed-set']['Recall'] = float(100.*recall_avg_auc.mean())
                result['Closed-set']['class-wise'] = list(100.*recall_avg_auc)
            else:
                raise RuntimeError("In Closed-set setting, `predict_labels_without_unknown` is indispensable.")
        else:
            # multi-class recall
            recall_avg_auc = recall_score(np_grds, np_prds, labels=np.unique(np_grds), average=None)
            # overall accuracy
            OA = np.mean(np_grds==np_prds)
            result['Open-set']['OA'] = float(100.*OA)
            result['Open-set']['OS'] = float(100.*recall_avg_auc.mean())
            result['Open-set']['class-wise'] = list(100.*recall_avg_auc)
            OS_star = recall_avg_auc[:-1].mean()
            result['Open-set']['OS*'] = float(100.*OS_star)
            UNK_score = recall_avg_auc[-1]
            result['Open-set']['UNK'] = float(100.*UNK_score)
            result['UniDA']['H-score'] = float(100.*2* OS_star*UNK_score / (OS_star + UNK_score + 1e-5))
            # calculate closed-set metrics on target-shared samples
            if len(predict_labels_without_unknown) > 0:
                prds_without_unknown = torch.cat(predict_labels_without_unknown)
                np_prds_without_ood = copy.copy(prds_without_unknown.numpy())
                acc_without_ood = np.mean(np_grds[target_shared_indexs]==np_prds_without_ood[target_shared_indexs])
                recall_avg_auc_without_ood = recall_score(np_grds[target_shared_indexs],\
                                                          np_prds_without_ood[target_shared_indexs],\
                                                          labels=np.unique(np_grds[target_shared_indexs]),\
                                                          average=None)
                result['Closed-set']['OA'] = float(100.*acc_without_ood)
                result['Closed-set']['Recall'] = float(100.*recall_avg_auc_without_ood.mean())
                result['Closed-set']['class-wise'] = list(100.*recall_avg_auc_without_ood)
            
        # NMI of target-private samples
        if len(feat) > 0:
            feat = torch.cat(feat)
            feat = F.normalize(feat)
            if n_target_private != 0:
                import time
                time_in = time.process_time()
                private_pred, _ = run_kmeans(feat[target_private_indexs], \
                                             n_target_private, init_centroids=None, gpu=True)  # gpu=True results in ERROR
                time_out = time.process_time()
                self.info.logger.debug(f"run_kmeans, time cost: {time_out - time_in:.4f}")
                nmi = normalized_mutual_info_score(gt_labels[target_private_indexs].numpy(), private_pred)
                result['Open-set']['NMI'] = float(100.*nmi)
                result['UniDA']['H3-score'] = float(100.*3*(1/(1/(OS_star+1e-5) +\
                                                               1/(UNK_score+1e-5) + \
                                                               1/(nmi+1e-5))))
        else:
            self.info.logger.info('not len(feat) > 0, skip the calculation of NMI.')

        if n_target_private != 0:
            np_prds_private = np_prds[target_private_indexs]
            np_prds_shared = np_prds[target_shared_indexs]

            TP = int((np_prds_shared != self.num_classes).sum())  # k -> k
            FN = int((np_prds_shared == self.num_classes).sum())  # k -> u
            FP = int((np_prds_private != self.num_classes).sum())  # u -> k
            TN = int((np_prds_private == self.num_classes).sum())  # u -> u

            result['Unknown_binary']['TP'] = TP
            result['Unknown_binary']['FN'] = FN
            result['Unknown_binary']['FP'] = FP
            result['Unknown_binary']['TN'] = TN

            if len(known_scores) > 0 and len(predict_labels_without_unknown) > 0:
                prds_without_unknown = torch.cat(predict_labels_without_unknown)
                np_prds_without_ood = copy.copy(prds_without_unknown.numpy())
                known_scores = torch.cat(known_scores).numpy()

                x1, x2 = known_scores[target_shared_indexs], known_scores[target_private_indexs]

                AUPR_score = average_precision_score([1] * len(x1) + [0] * len(x2), list(x1) + list(x2))
                AUPR_neg_score = average_precision_score([0] * len(x1) + [1] * len(x2), list(-x1) + list(-x2))
                # AUPR
                result['Unknown_binary']['AUPR'] = 100 * AUPR_score
                result['Unknown_binary']['AUPR-neg'] = 100 * AUPR_neg_score
                # AUROC
                result['Unknown_binary']['AUROC'] = \
                    100 * roc_auc_score(1 - np.array(target_private_indexs)*1, known_scores)
                # OSCR
                OSCR_socre, self.OSCR_ROC, self.OSCR_thr = compute_oscr(x1, x2, \
                        np_prds_without_ood[target_shared_indexs], np_grds[target_shared_indexs])
                
                result['UniDA']['UCR'] = 100 * OSCR_socre

        else:
            self.info.logger.info('UCR, Closed-set setting.')
            result['UniDA']['UCR'] = result['Closed-set']['OA']

        try:
            self.metrics[loader_ind] = result
        except:
            self.metrics.append(result)

    def save_metric(self, iter_num):
        log_dir = self.info.log_dir
        if not os.path.exists(log_dir+'/metrics'):
            os.makedirs(log_dir+'/metrics')
        for i in range(len(self.metrics)):
            save_dir = log_dir+f'/metrics/metric_{i}_{iter_num}.json'
            save_as_json(self.metrics[i], save_dir)
    

    def log_(self, iter_num, loss_dict):  
        print('(train) iter: {}, '.format(iter_num), end='')
        for _, key in enumerate(loss_dict):
            print('{0}: {1:.4f}, '.format(key, loss_dict[key]), end='')
        print('')
    
    def print_key_metric(self, iter_num, i, key_metric=None):
        if key_metric == None:
            key_metric ={'Closed-set':{  
                            'Recall': None,
                            'OA': None,
                        },
                        'Open-set':{  
                            'OS*': None,  # average classes accuracy of target-shared samples
                            'UNK': None,  # unk recall
                            'OS': None,  # average classes accuracy of (num_classes + 1) classes
                            'NMI': None,  # NMI of target-private samples
                        },
                        'Unknown_binary':{  # when known as pos
                            'AUROC': None,
                        },
                        'UniDA':{
                            'H-score': None,  # harmonic mean of OS* and UNK
                            'H3-score': None,  # harmonic mean of OS*, UNK and NMI
                            'UCR': None,  # UCR from paper uniood
                        }
                        }
        metrics = self.metrics[i]
        print('(val) iter: {}, '.format(iter_num))
        for key1 in metrics:
            if key1 in key_metric.keys():
                print(f"    {key1}: [ ", end='')
                for key2 in metrics[key1]:
                    if key2 in key_metric[key1].keys() and metrics[key1][key2] != None:
                        print(f"{key2}: {metrics[key1][key2]:.2f}, ", end='')
                print("]")

    # writer
    def writer_log_(self, iter_num, loss_dict):
        for _, key in enumerate(loss_dict):
            self.info.writer.add_scalar('loss/{}'.format(key), loss_dict[key], global_step=iter_num)
        if hasattr(self, 'time_in'):
            self.info.writer.add_scalar('time cost/forward', self.time_out - self.time_in, global_step=iter_num)
        if hasattr(self, 'time_data_in'):
            self.info.writer.add_scalar('time cost/data', self.time_data_out - self.time_data_in, global_step=iter_num)

    def writer_interval(self, iter_num):
        writer_dict = {
            'metrics': self.writer_metrics,
            'UCR-curve': self.writer_UCR_curve,
        }
        writer_ = self.writer_list()
        for writer_type in writer_:
            self.info.logger.info(f"writer interval {iter_num}: {writer_type} starting")
            writer_dict[writer_type](iter_num)

    def writer_list(self):
        return ['metrics', 'UCR-curve']

    def writer_metrics(self, iter_num, key_metric=None):
        if key_metric == None:
            key_metric ={'Closed-set':{  
                            'Recall': None,
                            'OA': None,
                        },
                        'Open-set':{  
                            'OS*': None,  # average classes accuracy of target-shared samples
                            'UNK': None,  # unk recall
                            'OS': None,  # average classes accuracy of (num_classes + 1) classes
                            'NMI': None,  # NMI of target-private samples
                        },
                        'Unknown_binary':{  # when known as pos
                            'TP': None,
                            'FN': None,
                            'FP': None,
                            'TN': None,
                            'AUROC': None,
                        },
                        'UniDA':{
                            'H-score': None,  # harmonic mean of OS* and UNK
                            'H3-score': None,  # harmonic mean of OS*, UNK and NMI
                            'UCR': None,  # UCR from paper uniood
                        }
                    }
        
        for loader_ind in range(len(self.metrics)):
            metrics = self.metrics[loader_ind]
            for key1 in metrics:
                if key1 in key_metric.keys():
                    for key2 in metrics[key1]:
                        if key2 in key_metric[key1].keys() and metrics[key1][key2] != None:
                            self.info.writer.add_scalar(
                                'Metrics-{}/{}_{}'.format(key1, key2, loader_ind), metrics[key1][key2], global_step=iter_num)

    def writer_UCR_curve(self, iter_num):

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        if  not hasattr(self, 'OSCR_ROC') or self.OSCR_ROC == None:
            self.info.logger.warning("self.OSCR_ROC == None, writer_UCR_curve failed.")
            return
        ROC = self.OSCR_ROC
        writer = self.info.writer

        # Compute AUROC Using Trapezoidal Rule
        OSCR = 0
        n = len(ROC) - 2
        for j in range(n+1):
            h =   ROC[j][0] - ROC[j+1][0]
            w =  (ROC[j][1] + ROC[j+1][1]) / 2.0
            OSCR = OSCR + h*w

        # roc curve
        fpr, tpr = zip(*ROC)
        plt.clf()
        plt.title('Receiver Operating Characteristic')
        temp = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.1f' % (OSCR*100))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Correct Classification Rate')
        plt.xlabel('False Positive Rate')

        # save image in Tensorboard
        fig = temp[0].get_figure()
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        writer.add_image('AUC/UCR-curve', image, global_step=iter_num, dataformats='HWC')
        plt.close()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CLIPLoRABase(nn.Module):
    
    def __init__(self, info: BasicInfo):
        super().__init__()
        self.info = info
        self.device = "cuda"
        self.iter_num = 0
        self.method_args = info.conf.method_args
        self.require_source = self.method_args.source
        self.require_target = self.method_args.target
        self.build_model()
        

    def build_model(self):
        cfg = self.info.conf.model
        self.lora_r = cfg.lora_r
        self.lora_r_t = cfg.lora_r_t

        # build visual model
        self.model_name = cfg.name
        self.clip_model = build_backbone_local(self.model_name)
        self.backbone = self.clip_model.visual
        self.text_encoder = LoRA_ViT(TextEncoder(self.clip_model), 
                                     lora_r=self.lora_r_t, 
                                     lora_alpha=self.lora_r_t)
        self.backbone = LoRA_ViT(self.backbone, 
                                 lora_r=self.lora_r, 
                                 lora_alpha=self.lora_r)
        self.clip_model_token_embedding = self.clip_model.token_embedding
        del self.clip_model 
        
        # determine the feature_dim
        if self.model_name in CLIP_MODELS:
            self.feature_dim = self.backbone.output_dim
        else:
            raise NotImplementedError()
        
        self.lab2cname, self.classnames_all, self.classnames_split = self.get_classnames()
        torch.cuda.empty_cache()
        

    def before_training(self):
        self.init_hyper_param()

        ## get large amounts of templates for better perfermance
        templates = get_templates(self.info.dataset_name, self.templates_type)
        n = max(1, int(self.templates_sampling_rate*len(templates)))
        templates = random.sample(templates, n)
        self.templates = np.char.array(templates)
        self.n_templates = len(self.templates)


    def init_hyper_param(self,):
        for key in self.method_args.keys():
            setattr(self, key, getattr(self.method_args, key))
            print(f"{key}: {getattr(self, key)}")


    def before_interval(self, iter_num):
        pass


    def custom_interval(self,):
        pass


    def before_forward(self, iter_num):
        self.train()
        self.iter_num = iter_num


    def get_classnames(self):

        cfg = self.info.conf.dataset
        _, task_target = get_task(cfg.name, cfg.task)
        class_split = make_class_split(cfg.name, cfg.shared, cfg.source_private, cfg.target_private)
        target_dataset = UniversalDataset(
                self.info.data_dir, 
                cfg.name, 
                task_target, 
                source=False,
                data_transforms=get_data_transforms('clip-none', 
                                                    self.info.conf.model.name), 
                class_split=class_split)
        
        return target_dataset.lab2cname, target_dataset.classnames_all, target_dataset.classnames_split
    

    @abstractmethod
    def forward(self, batched_inputs, lambda_dict):
        return

    def before_testing(self):
        self.eval()

    @abstractmethod
    def predict(self, batched_input):
        return
        
    def after_backward(self, iter_num):
        pass

    def after_interval(self, iter_num):
        # update, testing
        pass

    def after_training(self):
        # save
        pass


    @torch.no_grad()
    def embed_classnames(self, classnames, templates_type=None, need_token=True, templates_sampling_rate=None):

        self.before_testing()
        if templates_type is None:
            templates_type = self.templates_type

        ## get large amounts of templates for better perfermance
        templates = get_templates(self.info.dataset_name, templates_type)
        # Use sampling to reduce memory usage if necessary
        if templates_sampling_rate is None:
            templates_sampling_rate = self.info.conf.method_args.templates_sampling_rate
        ratio = templates_sampling_rate
        if ratio < 1.0:
            templates = random.sample(templates, max(1, int(ratio*len(templates))))
        self.templates = np.char.array(templates)

        ## preparing for forward propagation
        if need_token:
            embeddings_list = []
            tokenized_prompts_list = []
        text_features_list = []
        classnames = [name.replace("_", " ") for name in classnames]

        ## forward propagation
        for classname in classnames:
            prompts = [template.format(classname) for template in templates]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            embeddings = self.clip_model_token_embedding(tokenized_prompts)
            text_features = self.text_encoder(embeddings, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            if need_token:
                embeddings_list.append(embeddings)
                tokenized_prompts_list.append(tokenized_prompts)
            text_features_list.append(text_features)

        if need_token:
            embeddings = torch.stack(embeddings_list, dim=0).permute(1, 0, 2, 3)
            tokenized_prompts = torch.stack(tokenized_prompts_list, dim=0).permute(1, 0, 2)
            n_templates = embeddings.size(0)
        text_features = torch.stack(text_features_list, dim=0).detach()
        torch.cuda.empty_cache()
        
        if need_token:
            return text_features, embeddings, tokenized_prompts, n_templates
        else:
            return text_features
    