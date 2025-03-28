# Hardness-Aware Prototypical Contrastive Learning for Hyperspectral Coastal Wetlands Classification

## Usage



### 1. Setup
Install the dependency:
```bash
pip install -r requirements.txt --user
```

Create a `dataset`, `result` and `test_log` folder in the project directory:
```bash
mkdir dataset
mkdir result
mkdir test_log
```

### 2. Dataset prepare
Download our datasets and place them in `dataset` folder:

Google Drive: [https://drive.google.com/drive/folders/1jjg6Jlyb92pVrUzbdr5fHSMzYQnr2U47](https://drive.google.com/file/d/12eDOB99FE3MDU2MoO6GVHnT_M5YSMzli/view?usp=sharing)
│  

├── CongHoa

│   └── CongHoa.mat     # data

│   └── CongHoa_gt.mat  # ground truth

│   └── 200             # training set with 200 random samples per class
└── DongXing
    └── DongXing.mat     # data
    └── DongXing_gt.mat  # ground truth
    └── 20               # training set with 20 random samples per class

### 3. Unsupervised Pre-Training
We can run on a single machine of single GPU with the following command:
```
python main.py
```

### 4. Linear Classification
With a pre-trained model, we can easily evaluate its performance with:
```
python linear.py
```
