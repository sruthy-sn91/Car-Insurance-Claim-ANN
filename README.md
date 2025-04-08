# Car-Insurance-Claim-ANN

# Car Insurance Claim Prediction and Model Interpretability

This project focuses on predicting car insurance claims using a variety of machine learning models and exploring model interpretability using explainability techniques. The primary objective is to build and evaluate predictive models and then use interpretability tools like SHAP and LIME to gain insights into the feature importance and decision-making process of these models.

---

## Project Overview

The project follows these key steps:
1. **Data Ingestion and Preprocessing**:  
   - Importing and inspecting the dataset.
   - Handling missing values using imputation.
   - Data transformation: encoding categorical variables, feature scaling, and applying PCA for dimensionality reduction.
   - Balancing the dataset using SMOTE to address class imbalance.

2. **Exploratory Data Analysis (EDA)**:  
   - Univariate and bivariate analysis using visualizations (histograms, boxplots, and heatmaps).
   - Checking for skewness and handling outliers.

3. **Model Training and Evaluation**:  
   - Training multiple classification models including Logistic Regression, Random Forest, Gradient Boosting, SVM, Decision Tree, AdaBoost, XGBoost, K-Nearest Neighbors, and Gaussian Naive Bayes.
   - Applying hyperparameter tuning using GridSearchCV with StratifiedKFold.
   - Evaluating models using accuracy, precision, recall, F1 score, and ROC AUC metrics.

4. **ANN Modeling and Hyperparameter Experiments**:  
   - Building an Artificial Neural Network (ANN) with TensorFlow/Keras.
   - Conducting experiments to assess the impact of varying:
     - **Number of neurons** in the first layer (32, 64, 128).
     - **Number of layers** (1-layer, 2-layers, 3-layers architectures).
     - **Dropout rates** (0.0, 0.2, 0.3, 0.5).
     - **Batch size** (16, 32, 64).
   - Comparing the effect of these hyperparameters on model convergence and test accuracy.

5. **Explainability**:  
   - Using **SHAP** to understand the contribution of each feature for both Random Forest and ANN predictions.
   - Using **LIME** to provide local explanations for individual predictions and to compare model reasoning between the Random Forest and ANN.

---

## Libraries and Tools Used

- **Data Manipulation and Analysis**:  
  - `pandas`
  - `numpy`
  
- **Data Visualization**:  
  - `matplotlib`
  - `seaborn`

- **Data Preprocessing and Modeling**:  
  - `scikit-learn` (for imputation, encoding, scaling, PCA, model selection, and evaluation metrics)
  - `imblearn` (for SMOTE)
  - `xgboost`

- **Deep Learning**:  
  - `tensorflow` and `keras`

- **Explainability**:  
  - `shap`
  - `lime`

- **Other**:  
  - `scipy`
  - `statsmodels`

---

## Experiments and Results

### ANN Hyperparameter Experiments

- **Neurons in First Layer**:  
  Tested with 32, 64, and 128 neurons.  
  - Test accuracies improved from ~86.30% (32 neurons) to ~87.19% (128 neurons).

- **Model Architecture (Number of Layers)**:  
  Evaluated architectures with 1, 2, and 3 hidden layers.  
  - Deeper networks (e.g., 3-layer) generally achieved higher validation accuracies compared to shallower networks.

- **Dropout Rate**:  
  Evaluated dropout settings of 0.0, 0.2, 0.3, and 0.5.  
  - The results indicated that while dropout can help prevent overfitting, the differences in validation accuracy across these values were minor.

- **Batch Size**:  
  Compared batch sizes of 16, 32, and 64.  
  - A slight improvement in test accuracy was observed with an increasing batch size (from ~86.41% to ~86.74%).

### Interpretability Analysis

- **SHAP Analysis**:  
  - **Random Forest**: The SHAP summary plot typically shows a discrete, stepwise contribution of features, reflecting the threshold-based splits inherent in tree ensembles.
  - **ANN**: The SHAP plot for the neural network displays smoother, more continuous contributions, indicating gradual decision boundaries.
  
- **LIME Analysis**:  
  - For individual predictions, LIME highlighted which features push the prediction towards either “Claim” or “No Claim.”
  - The explanations differed between the Random Forest and ANN, with the Random Forest often showing a few dominating features and the ANN distributing importance more evenly across several inputs.

---

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/car-insurance-claim-prediction.git
   cd car-insurance-claim-prediction
