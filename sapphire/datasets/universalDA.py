
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# fix bug: https://blog.csdn.net/qq_35056292/article/details/117150457
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from torchvision.transforms import Compose
from mmcls.datasets.pipelines import Compose as MMCompose


def get_task(conf_dataset_name, conf_dataset_task):
    task_dict = {
        'office':{
            'a': 'amazon',
            'd': 'dslr',
            'w': 'webcam'
        },
        'officehome':{
            'A': 'Art',
            'C': 'Clipart',
            'P': 'Product',
            'R': 'Real'
        },
        'visda':{
            'S': 'train',
            'R': 'validation',
        },
        'domainnet':{
            'c': 'clipart',
            'i': 'infograph',
            'p': 'painting',
            'q': 'quickdraw',
            'r': 'real',
            's': 'sketch',
        },
        'CUB':{
            'S': 'train',
            'R': 'test',
        }
    }

    source = conf_dataset_task[0]
    target = conf_dataset_task[1]

    task_source = task_dict[conf_dataset_name][source]
    task_target = task_dict[conf_dataset_name][target]

    return task_source, task_target
    

def txt_parse(dataset_name, imgs):
    '''
    Parse the `classnames`(list) and `lab2cname`(dict) map of specific dataset.
    Note that, as a dict, the items in lab2cname are arranged in a certain order
    which depends on the txt file. This function sorts the items according to 
    the int(label), and then generates `classnames` from `lab2cname`, which ensures
    the `classnames` and `lab2cname` follow the same order depending on int(label).

    args:
        dataset_name: determine which dataset the txt file belongs to
        imgs: the content of txt file
    '''

    def sort_func(label_classname):
        return int(label_classname[0])
        
    import re
    lab2cname = {}

    if dataset_name == 'officehome':
        for x in imgs:
            classname = re.split(r'[/\ ]', x.strip())[1].replace("_", " ")
            label = int(re.split(r'[/\ ]', x.strip())[3])
            lab2cname[f'{label}'] = classname
        lab2cname_sorted = dict(sorted(lab2cname.items(), key=sort_func))
        classnames = list(lab2cname_sorted.values())
        return lab2cname_sorted, classnames

    elif dataset_name == 'office':
        for x in imgs:
            classname = re.split(r'[/\ ]', x.strip())[2].replace("_", " ")
            label = int(re.split(r'[/\ ]', x.strip())[4])
            lab2cname[f'{label}'] = classname
        lab2cname_sorted = dict(sorted(lab2cname.items(), key=sort_func))
        classnames = list(lab2cname_sorted.values())
        return lab2cname_sorted, classnames
    
    elif dataset_name == 'domainnet':
        for x in imgs:
            classname = re.split(r'[/\ ]', x.strip())[1].replace("_", " ")
            label = int(re.split(r'[/\ ]', x.strip())[3])
            lab2cname[f'{label}'] = classname
        lab2cname_sorted = dict(sorted(lab2cname.items(), key=sort_func))
        classnames = list(lab2cname_sorted.values())
        return lab2cname_sorted, classnames
    
    elif dataset_name == 'visda':
        for x in imgs:
            classname = re.split(r'[/\ ]', x.strip())[1].replace("_", " ")
            label = int(re.split(r'[/\ ]', x.strip())[3])
            lab2cname[f'{label}'] = classname
        lab2cname_sorted = dict(sorted(lab2cname.items(), key=sort_func))
        classnames = list(lab2cname_sorted.values())
        return lab2cname_sorted, classnames
    
    elif dataset_name == 'CUB':
        for x in imgs:
            classname = re.split(r'[/\ .]', x.strip())[2].replace('_', ' ')
            label = int(re.split(r'[/\ .]', x.strip())[-1])
            lab2cname[f'{label}'] = classname
        lab2cname_sorted = dict(sorted(lab2cname.items(), key=sort_func))
        classnames = list(lab2cname_sorted.values())
        return lab2cname_sorted, classnames

    else:
        raise NotImplementedError(f"txt_parse() of dataset {dataset_name} is not Implemented!")


def make_class_split(dataset_name, shared, source_private, target_private):
    num_all = {
        'office': 31,
        'officehome': 65,
        'domainnet': 345,
        'visda': 12,
        'CUB': 200,
    }
    num_class = num_all[dataset_name]
    if num_class > shared + source_private + target_private:
        print("WARNING: shared/source_private/target_private = {}/{}/{}. Not all categories in datasets are used!"
                .format(shared, source_private, target_private))
    elif num_class < shared + source_private + target_private:
        raise RuntimeError("shared/source_private/target_private = {}/{}/{}. total number of splits > number of class in dataset {}"
                .format(shared, source_private, target_private, dataset_name, num_all))

    class_split = {}
    class_split['shared'] = list(range(shared))
    class_split['source_private'] = list(range(shared, shared+source_private))
    class_split['target_private'] = list(range(shared+source_private, shared+source_private+target_private))
    if dataset_name == 'office' and source_private == 0 and target_private != 0:
        class_split['target_private'] = list(range(num_class-target_private, num_class))

    return class_split


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')




class UniversalDataset(Dataset):
    '''
    Implementation of datasets used in universal DA. Some code are redundant
    for scalability, and will be completed in the future.

    Args:
        data_root: The root path of all data, such as '/data/user/Dataset'.
        dataset_name: The name of dataset. Could be 'office', 'officehome', 'visda' and 'domainnet'.
        domain: Indicate the domain label, such as 'Art' and 'Clipart' in officehome.
        source: Indicate which domain.
        class_split: A dict with following format.
            {'shared': list, 'source_private': list, 'target_private': list}
    '''
    def __init__(self, data_root, dataset_name, domain,
                source=True, data_transforms=None, class_split=None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.domain = domain
        self.source = source
        self.transforms = data_transforms
        self.num_class = len(class_split['shared']+class_split['source_private']+class_split['target_private'])

        if class_split == None:
            raise Exception("'class_split' is None!")

        self.class_shared = class_split['shared']
        self.class_source_private = class_split['source_private']
        self.class_target_private = class_split['target_private']

        path = os.path.join(self.data_root, dataset_name)
        image_list = os.path.join(self.data_root, 'txt', self.dataset_name, self.domain+'.txt')

        if self.source == True:
            imgs, labels = self.make_uni_dataset_fromlist(path, image_list, self.class_shared, 
                                self.class_source_private)
        elif self.source == False:
            imgs, labels = self.make_uni_dataset_fromlist(path, image_list, self.class_shared, 
                                self.class_target_private)
            
        # get classnames and a dict mapping label to classname
        self.get_classnames(path, image_list, class_split)

        # split the classnames into 3 parts
        self.get_classnames_split()

        self.imgs = imgs
        self.labels = labels
        self.loader = rgb_loader
        
        num_imgs = self.split_count(labels, class_split)
        self.num_imgs_shared = num_imgs[0]
        self.num_imgs_src_pri = num_imgs[1]
        self.num_imgs_tar_pri = num_imgs[2]


    def load_image(self, path, transforms):

        mmcv_flag = True if type(transforms) is MMCompose else False

        if mmcv_flag:
            data = dict(img_info=dict(filename=path), gt_label=0, img_prefix=None)
            img = transforms(data)['img']
        else:
            img = rgb_loader(path)
            img = transforms(img)

        return img


    def __getitem__(self, index):
        path = self.imgs[index]
        gt_label = self.labels[index]
        data_info = {}

        img = self.load_image(path, self.transforms[0])

        # Strong Augmentations
        if len(self.transforms) > 1:
            aug = []
            for transforms in self.transforms[1:]:
                temp = self.load_image(path, transforms)
                aug.append(temp)

        data_info['filename'] = path
        data_info['image_ind'] = index


        if len(self.transforms) > 1:
            return {'img': img, 
                    'aug': aug,
                    'gt_label': gt_label, 
                    'data_info': data_info
                    }
        else:
            return {'img': img, 
                    'gt_label': gt_label, 
                    'data_info': data_info
                    }


    def __len__(self):
        return len(self.imgs)
    

    def make_uni_dataset_fromlist(self, path, image_list, shared, private):
        '''
        Args:
            path: e.g. path to 'officehome' dataset
            image_list: path to txt file.
            shared: list of shared categories in universal DA.
            private: list of private categories in universal DA.
        '''
        all_categories = shared + private
        with open(image_list) as f:
            imgs = [path + '/' + x.split(' ')[0] 
                        for x in f.readlines() 
                        if int(x.split(' ')[1].strip()) in all_categories]
        with open(image_list) as f:
            labels = []
            for ind, x in enumerate(f.readlines()):
                label = x.split(' ')[1].strip()
                if int(label) in all_categories:
                    labels.append(int(label))
        return imgs, labels
    

    def split_count(self, labels, class_split):
        shared = 0
        src_pri = 0
        tar_pri = 0
        for i in range(len(labels)):
            if labels[i] in self.class_shared:
                shared += 1
            elif labels[i] in self.class_source_private:
                src_pri += 1
            elif labels[i] in self.class_target_private:
                tar_pri += 1
        return [shared, src_pri, tar_pri]
    

    def get_classnames(self, path, image_list, class_split):
        all_categories = class_split['shared'] + \
                         class_split['source_private'] + \
                         class_split['target_private']
        with open(image_list) as f:
            imgs = [x for x in f.readlines() 
                      if int(x.split(' ')[1].strip()) in all_categories]
        self.lab2cname, self.classnames_all = txt_parse(self.dataset_name, imgs)


    def get_classnames_split(self):
        self.classnames_split = {}
        self.classnames_split['shared'] = \
            [ self.lab2cname[str(i)] for i in self.class_shared]
        self.classnames_split['source_private'] = \
            [ self.lab2cname[str(i)] for i in self.class_source_private]
        self.classnames_split['source'] = \
            [ self.lab2cname[str(i)] for i in self.class_shared+self.class_source_private]
        
        # Execute the conditional statement because there are missing samples 
        # for class No.327("t-shirt") in the "painting" domain of the cleaned 
        # DomainNet dataset.
        if self.dataset_name == 'domainnet':
            import warnings
            warnings.warn('''The dataset DomainNet is missing samples from certaincategories. Thus, we couldn't collect the classnames_split of 'target_private and 'target'.''')
        
        else:
            self.classnames_split['target_private'] = \
                [ self.lab2cname[str(i)] for i in self.class_target_private]
            self.classnames_split['target'] = \
                [ self.lab2cname[str(i)] for i in self.class_shared+self.class_target_private]
            


class DATALoader(DataLoader):

    '''
    Copy from https://blog.csdn.net/a237072751/article/details/124599426
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

