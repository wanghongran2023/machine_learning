# Pneumonia Detection Using Chest X-Rays

This project focuses on building an effective machine learning solution to detect pneumonia from chest X-ray images. The project uses advanced deep learning models, including ResNet-34 and DenseNet-121, and employs AWS SageMaker for scalable and efficient training.

## Table of Contents
1. [Introduction](#introduction)
2. [Software and Libraries Used](#software-and-libraries-used)
3. [Dataset Information](#dataset-information)
4. [Setup Instructions](#setup-instructions)
5. [Project Workflow](#project-workflow)
6. [Results](#results)

---

## Introduction
Pneumonia is a life-threatening lung infection, and timely diagnosis is crucial for effective treatment. This project aims to automate pneumonia detection by training a machine learning model on labeled chest X-ray datasets. By leveraging ResNet-34 and DenseNet-121 in an ensemble approach, the project aspires to provide a scalable, cost-effective diagnostic solution.

## Software and Libraries Used
The project uses the following software and libraries:

### Core Frameworks
- **AWS SageMaker**: To run scalable machine learning workflows and train models on the cloud.
- **PyTorch (v1.5.0)**: For deep learning model implementation.
- **Kaggle Hub**: To download datasets directly from Kaggle.

### Supporting Libraries
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Pillow**: For image processing.
- **Boto3**: AWS SDK for Python to interact with S3 buckets.

### Installation
Ensure the following Python packages are installed:
```bash
pip install sagemaker
pip install kagglehub
```

## Dataset Information
The dataset used for this project is the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). It consists of labeled images categorized as `Normal` and `Pneumonia`. The dataset is divided into training, testing, and validation sets.

- Training Set: Used for model training.
- Validation Set: Used for hyperparameter tuning.
- Testing Set: Used for final evaluation.

## Setup Instructions
1. **Environment Setup**:
   - Use AWS SageMaker or a local Python environment.
   - Configure AWS CLI with appropriate permissions to interact with S3.

2. **Dataset Download**:
   - Use the Kaggle Hub library to download the dataset:
     ```python
     import kagglehub
     path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
     ```

3. **Data Preparation**:
   - Organize the dataset into `train`, `test`, and `validation` folders.
   - Upload the data to an S3 bucket for use in SageMaker.

4. **Model Training**:
   - Use the `PyTorch` Estimator in SageMaker to train the models (ResNet-34 and DenseNet-121).
   - Update the `train_script.py` or `train_script_combined2.py` with appropriate hyperparameters.

5. **Evaluation**:
   - Use the validation set to evaluate model performance.
   - Analyze metrics like accuracy, precision, recall, and F1 score.

## Project Workflow
1. **Setup and Initialization**:
   - Install required libraries.
   - Create SageMaker session and retrieve IAM roles.

2. **Dataset Exploration and Visualization**:
   - Download, explore, and visualize the dataset.
   - Summarize the dataset and plot class distributions.

3. **Model Training**:
   - Train ResNet-34 as a benchmark model.
   - Train ResNet and DenseNet separately and evaluate their performance.
   - Combine ResNet and DenseNet models using weighted averaging for ensemble learning.

4. **Performance Comparison**:
   - Compare results of ResNet, DenseNet, and their ensemble.
   - Visualize metrics to determine the best-performing model.

## Results
- ResNet-34 trained for 10 epochs achieved a validation accuracy of **83.17%**, outperforming the ensemble approach.
- DenseNet-121 converged faster with higher training accuracy but showed less generalization compared to ResNet.
- Combined models (ResNet + DenseNet) did not outperform ResNet alone, emphasizing the latter’s robustness for this task.

## References
1. [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
3. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
5. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks.

