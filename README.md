# 1. Introduction of Business Problem
The aim of our project is to estimate whether a bank's customers leave the bank or not. As we know, it is much more expensive to sign in a new client than to keep an existing one. It is advantageous for banks to know what leads a client toward the decision to leave the company. The event that defines customer abandonment is the closing of the customer's bank account. Churn prevention allows companies to develop loyalty programs and retention campaigns to keep as many customers as possible. 

# 2. Data Understanding
## 2.1 Data Set Story
The data set consists of 10000 observations and 12 variables. Independent variables contain information about customers. The dependent variable refers to customer abandonment.
## 2.2 Explaintory Data Analysis (EDA) and Data Preprocess
We conducted thorough data preprocessing to improve the dataset's quality and distribution. This involved addressing missing values, skewness, outliers, and class imbalance. Meaningless columns ('RowNumber', 'CustomerId', 'Surname') were dropped. Encoding 'Geography' and 'Gender' transformed raw data for better machine learning algorithm processing. We used oversampling to balance the dataset without losing information. Finally, data standardization ensured comparability among variables, mitigating bias in feature importance and model performance due to differing scales.

# 3. Feature Engineering and Selection
Feature engineering plays a crucial role in enhancing model performance and extracting valuable insights from the data.
## 3.1 Pairplot Analysis
In order to visualize the relationships between pairs of features, we conducted a pairplot analysis. This technique provides a comprehensive view of the interactions and correlations among the selected features. But the resulting pairplot did not reveal many noteworthy observations. 
## 3.2 Principal Component Analysis (PCA)
To address the challenge of high-dimensional data, we applied Principal Component Analysis (PCA), a popular dimensionality reduction technique. By transforming the features into a new set of uncorrelated variables called principal components, PCA helped us capture the most important information while minimizing redundancy. The results of PCA revealed that 54% of the variance could be explained by the first five principal components.
## 3.3 Feature Importance
Determining feature importance is crucial for identifying the most influential factors in predicting customer churn. We utilized Random Forest to compute feature importance scores. The analysis revealed that customer age, balance, isActiveMember, EstimatedSalary and CreditScore were the top-ranked features in terms of importance. These findings align with our domain knowledge and provide valuable insights into the factors that significantly impact customer churn.
## 3.4 Recursive Feature Elimination (RFE)
To further refine the feature set, we employed Recursive Feature Elimination (RFE), a technique that recursively removes less important features. By iteratively eliminating the least significant features, RFE helped us identify the subset of features that contributed the most to the model's performance. Based on this analysis, we selected a set of features consisting of CreditScore, Age, Balance, NumOfProducts, and EstimatedSalary. In further classification, we do not eliminate other features and reserve all features to fit the model. If there is a sign of overfitting, we will back here and use this result to choose the features.

# 4. Classification
## 4.1 Low Code - PyCaret
We first split data into training and test data and then we used PyCaret library for a low-code machine learning pipeline to compare multiple machine learning models and select the best performing model as the best model. Here, the AUC (Area Under the Curve) indicator is used to evaluate the performance of the model, and sort='AUC' specifies to sort by the AUC value. After this, the code performed grid search tuning on the best model to further optimize the performance of the mode and evaluate the performance of the tuned model. 
## 4.2 Classifiers
In this part, we will use the classifiers, including Logistic Regression, SVM, K Nearest Neighbors (KNN), Random Forest, XGBoost, Gaussian Naive Bayes, Decision Tree, Multi-layer Perceptron, in the Scikit-learn library to train and evaluate multiple classification models, and use cross-validation to calculate the accuracy, balance accuracy, and AUC value of the model. 
According to the result, the XGBoost performed best with a balanced accuracy of 0.867 and the mean AUC of 0.939. Then we plot the learning curve for this classifier, the training and cross validation score showed a convergence trend, which means that there is no overfitting problem and the classifier can be accepted with great performance.
## 4.3 PCA Classification
By using PCA for feature dimensionality reduction, the high-dimensional original feature space can be transformed into a low-dimensional principal component space. The purpose of this is to reduce the dimensionality of features and possibly remove some noise or redundant information, thereby improving the performance of the model. Using PCA-posted data for training and evaluation can reduce computational cost and, in some cases, improve model performance. The result showed that the SVM classifier wins with a balanced accuracy of 0.845 and the mean AUC of 0.91. Also, the graph of the learning curve showed great training and testing scores and trends.
## 4.4 Grid Search
According to the results of classifications on the original data and PCA classification, the XGBoost classifier on the original data performs best, so we choose this model to do the grid search. The purpose of grid search is to find the best hyperparameters for the XGBoost model, and to train and evaluate the model based on the best parameters found. After grid searching, we found best parameters and implemented this improved model on the data. The balanced accuracy and the mean AUC are improved to 0.897 and 0.945.

# 5. Interpretation of Model (XAI)
## 5.1 Similar Model and Surrogate Model
In feature importance analysis, "similar model" and "surrogate model" are two different concepts and methods. Similar Model refers to using algorithms or models similar to the main model for feature importance analysis. It helps understand feature contributions by using models with similar structure and properties. Surrogate Model approximates the original model using a simplified version to better understand feature importance. It has a simpler structure and fewer features, making analysis easier. Surrogate models can be decision trees, linear models. Both methods aid in interpreting feature importance and can guide feature selection, model optimization, or problem understanding. 
In our feature importance analysis for these 2 models, the results are similar. Variables like ‘Age’, ‘Balance’, ‘IsActiveMember’’ are common, while ‘NoOfProducts’ is also important in the random forest model and ‘EstimatedSalary’ counts a lot in the surrogate model. 
## 5.2 Shapley Values and Partial Dependence Plots
Then we use Shapley Values to explain the importance of features predicted by a model. It quantifies the importance of features by calculating how much each feature contributes to the model output. Combining the result of shapley values and the feature importance analysis from the models, we decided to use ‘Age’, ‘IsActiveMember’ a ‘NoOfProducts’ in partial dependence plots. The result did showed some patterns:
Age is positively correlated with churn likelihood
Customer activity is negatively correlated with churn likelihood
The relationship between NoOfProducts and customer churn needs to be discussed on a case-by-case basis: when it is less than 1, it is negatively correlated; between 1-2.5, it shows a high positive correlation, which means that the increase in the number of products will greatly increase the possibility of loss; once it exceeds 2.5, the number An increase in will slightly increase the chance of churn.
Finally, we used decision tree as the surrogate model and the result showed that our feature importance analysis reasonable because the different branches of the decision tree represent the range of different values of a certain variable, the prediction results of the decision tree are almost consistent with the impact of the previously analyzed variables on the results.

# 6. Implement of Automated Machine Learning (AutoML)
In addition to the feature engineering and model development processes, we leveraged the power of Automated Machine Learning (AutoML) techniques to further enhance the predictive capabilities of our models.
AutoML offers a systematic and efficient approach to model selection, hyperparameter tuning, and ensemble creation. It automates the labor-intensive tasks involved in model development, allowing us to rapidly iterate through multiple algorithms and configurations.
To implement AutoML, we utilized Auto-Sklearn.
The outputs of AutoML shows it runs 5 target algorithms and 3 of them successed. Among them, the best validation score is 0.887515. The AUC of AutoML model is 0.875 and the accuracy is 0.875, which a little bit worse compared to our best XGBoost classifier.

# 7.Conclusion and Recommendations
Based on the findings of our project, we recommend the following strategies to address customer churn:
Personalized Retention Programs: Leverage the model's predictions to design personalized retention programs for customers identified as having a high likelihood of churn. 
Customer Segmentation: Utilize the model's insights to segment the customer base into different categories based on their churn probability.
Continuous Model Monitoring and Improvement: Regularly evaluate the performance of the machine learning model and update it as new data becomes available. Continuous monitoring allows for model refinement, ensuring its effectiveness in an evolving banking landscape.
In conclusion, our big data machine learning project successfully addressed the business problem of estimating customer churn in the banking industry. Through feature engineering techniques, we identified crucial factors that significantly impact customer churn, such as customer age, balance, and is active member.
The developed machine learning model provides accurate predictions and insights into customer churn, enabling the bank to make informed business decisions and implement effective retention strategies.
