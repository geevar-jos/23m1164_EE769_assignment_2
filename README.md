# 23m1164_EE769_assignment_2
Assignment 2 for EE769 Introduction to Machine Learning (Spring 2023-24)

# Machine Learning Models for Regression, Classification & Feature Extraction

## Objective
To explore and compare machine learning models for **regression**, **classification**, and **feature extraction** tasks, and to evaluate their performance, generalization ability, and deployment.

---

## Datasets
1. **Wine Quality Dataset** (UCI) [Dataset Link](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
   - Red & White wine data (≈6k samples, 11 features).  
   - Target: Wine quality (0–10 scale).  

2. **Mice Protein Expression Dataset** (UCI)  [Dataset Link](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)
   - Expression levels of 77 proteins (≈1k samples).  
   - Target: Genotype classification (binary).  

3. **Image Dataset (ResNet18 feature extraction)**  [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
   - Pre-trained ResNet18 used as a **fixed feature extractor**.  
   - Generated **Nx512 embeddings** for downstream classification.  

---

## Key Questions
1. How do **regression models** perform on predicting wine quality, and do they generalize across red/white wines?  
2. Can **classification models** predict Down syndrome genotype from protein expression data?  
3. How can **pre-trained CNNs (ResNet18)** be used to extract domain-specific features for classification?  
4. Can a trained model be **deployed** for real-time predictions via a web app?  

---

## Methods & Analysis
- **Data Preprocessing**
  - Cleaned and normalized datasets.  
  - Handled missing values in Mice Protein dataset using **Multivariate Feature Imputation**.  

- **Regression Models (Wine Quality)**
  - Trained **Random Forest, SVR (RBF), Neural Net (1 hidden layer)**.  
  - Evaluated with **R² and RMSE**.  
  - Tested out-of-distribution (red ↔ white wine predictions).  

- **Classification Models (Mice Protein)**
  - Trained **Random Forest, SVM (RBF), Neural Net (softmax output)**.  
  - Applied **RFECV (Recursive Feature Elimination)** for feature selection.  
  - Evaluated with **Accuracy & F1-score**.  

- **Feature Extraction with CNNs**
  - Used **ResNet18 pre-trained on ImageNet** as fixed feature extractor.  
  - Extracted **512-D embeddings** for images.  
  - Compared **Random Forest vs. RBF SVM** classifiers.  

- **Model Deployment**
  - Deployed regression model via **Streamlit web app**.  
  - Built GUI sliders for acidity, alcohol, sulphates, etc.  

---

## Results & Insights
- **Wine Regression**
  - Random Forest performed best (**R² ≈ 0.55, RMSE ≈ 0.63**).  
  - **Key predictors**: Alcohol, sulphates, volatile acidity.  
  - **Poor generalization** across red ↔ white wines (R² < 0.20).  

- **Mice Classification**
  - SVM with RBF kernel achieved **Accuracy ≈ 88%, F1 ≈ 0.86**, outperforming RF and NN by ~5–7%.  
  - **RFECV reduced features by ≈30%**, improving accuracy by ~3%.  

- **CNN Feature Extraction**
  - Extracted **512-D embeddings** from ResNet18.  
  - **RBF SVM reached Accuracy ≈ 90%, F1 ≈ 0.89**, ~6% higher than Random Forest.  

- **Deployment**
  - Streamlit web app enabled **real-time wine quality prediction** with user-friendly sliders.  

---

## Key Highlights
- **Profiled** 6k+ wine samples, trained & compared regression models → **Random Forest best with R² ≈ 0.55**.  
- **Processed** 1k+ protein samples with imputation → **SVM reached ≈88% accuracy, F1 ≈ 0.86**.  
- **Extracted** ResNet18 embeddings (512-D) → **SVM achieved ≈90% accuracy, F1 ≈ 0.89**.  
- **Deployed** interactive **Streamlit app** for real-time wine quality prediction.  

---

## Tools & Libraries
- **scikit-learn** – Random Forest, SVM, RFECV  
- **TensorFlow / Keras** – Neural Network models  
- **PyTorch** – ResNet18 feature extraction  
- **Pandas, NumPy, Seaborn, Matplotlib** – Data preprocessing & visualization  
- **Streamlit** – Model deployment with GUI  

---

## ✅ Conclusion
This project demonstrates an **end-to-end ML pipeline**:  
- From **data preprocessing & model training**,  
- To **feature extraction using pre-trained CNNs**,  
- And **deployment as an interactive web app**.  

It highlights skills in **regression, classification, feature selection, transfer learning, and model deployment** with measurable performance improvements.  

---

