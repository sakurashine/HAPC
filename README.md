# Hardness-Aware Prototypical Contrastive Learning for Hyperspectral Coastal Wetlands Classification

## Usage

### 1. Dataset prepare
Download our datasets then place them in data folder

Google Drive: [https://drive.google.com/drive/folders/1jjg6Jlyb92pVrUzbdr5fHSMzYQnr2U47](https://drive.google.com/file/d/12eDOB99FE3MDU2MoO6GVHnT_M5YSMzli/view?usp=sharing)

### 2. Setup
Before running the code, create a `result`  and a `test_log` folder in the project directory:
```bash
mkdir result
mkdir test_log

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
