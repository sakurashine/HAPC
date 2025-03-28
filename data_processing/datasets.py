import random
import torch
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
import h5py
import random

DATASETS_CONFIG = { # dict{'key':'value'}
    'PaviaU': { 
        'img': 'PaviaU.mat',   # 图片文件
        'gt': 'PaviaU_gt.mat'  # 标签文件
    },
    'KSC': {
        'img': 'KSC.mat',
        'gt': 'KSC_gt.mat'
    },
    'IndianPines': {
        'img': 'indian_pines.mat',
        'gt': 'indian_pines_gt.mat'
    },
    'Salinas': {
        'img': 'salinas.mat',
        'gt': 'salinas_gt.mat'
    },
    'CongHoa': {
        'img': 'CongHoa.mat',
        'gt': 'CongHoa_gt.mat'
    },
    'DongXing': {
        'img': 'DongXing.mat',
        'gt': 'DongXing_gt.mat'
    }
}

def get_dataset(dataset_name, target_folder, datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    #获取数据集路径
    img_file = target_folder + '/' + datasets[dataset_name].get('img')
    gt_file = target_folder + '/' + datasets[dataset_name].get('gt')

    #因为每种数据集的标签各不相同，比如印第安松林有16个类别，故label_values保存为17个值的数组，Undefined表示不属于任何类（黑点）
    if dataset_name == 'PaviaU':
        img = loadmat(img_file)['paviaU']
        gt = loadmat(gt_file)['Data_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]
    elif dataset_name == 'IndianPines':
        img = loadmat(img_file)
        img = img['HSI_original']
        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]
    elif dataset_name == 'KSC':
        img = loadmat(img_file)['KSC']
        gt = loadmat(gt_file)['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        ignored_labels = [0]
    elif dataset_name == 'Salinas':
        img = loadmat(img_file)['HSI_original']
        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Brocoli green weeds 1", "Brocoli_green_weeds_2",
                        "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble",
                        "Celery", "Grapes untrained", "Soil vinyard develop",
                        "Corn senesced green weeds", "Lettuce romaine 4wk",
                        "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
                        "Vinyard untrained", "Vinyard vertical trellis"]
        ignored_labels = [0]
    elif dataset_name == 'CongHoa': 
        # 数据集形状： (1360, 955, 32)
        # 类别：[ 0  1  2  3  4  5  6  7  8  9 ]
        # 训练集每类个数[386875, 447796, 9365, 28347, 123941, 3950, 122771, 83846, 79400, 12230] 
        # 总计911925个可用样本
        img = loadmat(img_file)['area1']
        gt = loadmat(gt_file)['labels']  # label1
        label_values = ["Undefined", "Forest", "Impervious",
                        "Bare", "River", "Lake",
                        "Coastal swamp", "Coastal marsh", "Rice", "Salt pans"]
        ignored_labels = [0]
    elif dataset_name == 'DongXing':  
        # 数据集形状： (945, 1274, 32)
        # 类别：[ 0  1  2  3  4  5  6  7  8  9 10 11 12]
        # 训练集每类个数[1036785, 25297, 1573, 11767, 2469, 6172, 5980, 629, 20950, 65758, 835, 23477, 2238]
        # 总计167145个可用样本
        img = loadmat(img_file)['19_image']
        gt = loadmat(gt_file)['19_label']
        img = np.transpose(img, (1, 2, 0))
        label_values = ["Undefined", "Forest", "Impervious",
                        "Brick house", "Bare", "River", "Lake",
                        "Coastal swamp", "Coastal marsh", "Salt pans",
                        "Tidal flat", "Aquaculture",
                        "Reservoir"]
        ignored_labels = [0]

    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        logger.info("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
              "data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    print("数据集形状：",np.shape(img))
    ignored_labels = list(set(ignored_labels))
    img = np.asarray(img, dtype='float32')
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    data = preprocessing.minmax_scale(data, axis=1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data = scaler.fit_transform(data)
    img = data.reshape(img.shape) 
    
    return img, gt, label_values, ignored_labels

class Hyper2X(torch.utils.data.Dataset):
    # 处理高光谱数据的一个工具包，增加了正样本构造功能
    """ Generic class for a hyperspectral scene """
    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(Hyper2X, self).__init__()
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.mixup_augmentation = hyperparams['mixup_augmentation']
        self.mixup_ratio = hyperparams['mixup_ratio']
        # self.mixup_neighbor = hyperparams['mixup_neighbor']
        self.mixup_neighbor = random.randrange(1, 15)
        if hyperparams['flip_augmentation']:
            data_copy = data.copy()
            gt_copy = gt.copy()
            for i in range(1): 
                data = np.hstack((data, data_copy))
                gt = np.hstack((gt, gt_copy))
        self.data = data
        self.label = gt
        self.classes = hyperparams['n_classes']
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.center_pixel = hyperparams['center_pixel']
        self.center_pixel = True
        supervision = hyperparams['supervision']

        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        # 丢弃了那些图像边缘的点（以所取像素为中心点，若该patch大小覆盖了图像边缘之外的部分，该像素丢弃）
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)
        self.count = len(self.labels) // 2
    
    @staticmethod
    def ud_flip(data, label):
        data = np.flipud(data)
        label = np.flipud(label)
        return data, label

    @staticmethod
    def lr_flip(data, label):
        data = np.fliplr(data)
        label = np.fliplr(label)
        return data, label

    @staticmethod
    def trans_flip(data, label):
        data = data.transpose((1, 0, 2))
        label = label.transpose((1, 0))
        return data, label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data0 = self.data[x1:x2, y1:y2]
        # 取空间领域patch进行SpectraMix增强以构造正样本
        data0_B = self.data[x1+self.mixup_neighbor:x2+self.mixup_neighbor, y1+self.mixup_neighbor:y2+self.mixup_neighbor]
        label0 = self.label[x1:x2, y1:y2]

        idx = i + 1
        if idx >= len(self.indices):
            idx = i - 1
        w, z = self.indices[idx]
        w1, z1 = w - self.patch_size // 2, z - self.patch_size // 2
        w2, z2 = w1 + self.patch_size, z1 + self.patch_size
        data1 = self.data[x1:x2, y1:y2]
        label1 = self.label[x1:x2, y1:y2]

        # 选择是否需要旋转增强
        if self.flip_augmentation:
            if (i > self.count) & (i <= self.count * 2):
                data1, label1 = self.ud_flip(data0, label0)
            elif (i > self.count * 2) & (i <= self.count * 3):
                data1, label1 = self.lr_flip(data0, label0)
            elif i > self.count * 3:
                data1, label1 = self.trans_flip(data0, label0)

        data0 = np.asarray(
            np.copy(data0).transpose(
                (2, 0, 1)), dtype='float32')
        label0 = np.asarray(np.copy(label0), dtype='int64')
        data1 = np.asarray(
            np.copy(data1).transpose(
                (2, 1, 0)), dtype='float32')
        label1 = np.asarray(np.copy(label1), dtype='int64')

        data0 = torch.from_numpy(data0)
        label0 = torch.from_numpy(label0)
        data1 = torch.from_numpy(data1)
        label1 = torch.from_numpy(label1)

        data0_B = np.asarray(
            np.copy(data0).transpose(
                (0, 1, 2)), dtype='float32')
        data0_B = torch.from_numpy(data0_B)
        # 取出中心点像素的标签
        if self.center_pixel and self.patch_size > 1:
            label0 = label0[self.patch_size // 2, self.patch_size // 2]
            label1 = label1[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data0 = data0[:, 0, 0]
            label0 = label0[0, 0]
            data1 = data1[:, 0, 0]
            label1 = label1[0, 0]
        # torch.Size([103, 11, 11]), torch.Size([]), torch.Size([103, 11, 11]), torch.Size([])

        # SpectraMix增强
        if self.mixup_augmentation:
            lambda_value = self.mixup_ratio
            # lambda_value = 0.2
            data1 = np.zeros_like(data0)
            for c in range(data0.shape[0]):  # Iterate over each spectral band
                data1[c] = lambda_value * data0[c] + (1 - lambda_value) * data0_B[c]
            # data1 = lambda_value * data0 + (1 - lambda_value) * data0_B

        return data0, label0, data1, label1

class HyperX(torch.utils.data.Dataset):
    # 处理高光谱数据的一个工具包
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.classes = hyperparams['n_classes']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.center_pixel = hyperparams['center_pixel']
        self.center_pixel = True
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0 
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        # 丢弃了那些图像边缘的点（以所取像素为中心点，若该patch大小覆盖了图像边缘之外的部分，该像素丢弃）
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        data = torch.from_numpy(data)
        if data_3D:
            data = data.unsqueeze(0) 
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        return data, label