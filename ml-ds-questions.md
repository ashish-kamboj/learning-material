### Q. Difference between Statistical Modeling and Machine Learning

---

### Q. What is selection Bias?
Selection Bias is a clinical trials is a result of the sample group not representing the entire target population.

---

### Q. Probability vs Likelihood [(here)](https://medium.com/swlh/probability-vs-likelihood-cdac534bf523)
  - Probability is used to finding the chance of occurrence of a particular situation, whereas Likelihood is used to generally maximizing the chances of a particular situation to occur.
  - [What is the difference between “likelihood” and “probability”?](https://stats.stackexchange.com/questions/2641/what-is-the-difference-between-likelihood-and-probability)

---

### Q. Types of Distribution [(here)](https://www.analyticsvidhya.com/blog/2017/09/6-probability-distributions-data-science/)

---

### Q. What is normal distribution

---

### Q. Baye's Theorem** [(here)](https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification)
  - Independent Events
  - Dependent Events
  -   - conditional Probability  :: P(A|B) = P(A∩B)/P(B)
					<br>OR</br>
			                          P(B|A) = P(B∩A)/P(A)	
  - Baye's Theorem :: P(A|B) = P(B|A)*P(A)/P(B)	<br>
		   P(A|B) = Posterior Probability <br>
		   P(B|A) = Likelihood <br>
		   P(A) = Prior probability <br>
		   P(B) = Marginal Probability
---	

### Q. Covariance Vs Correlation
  - **Covariance** - indicates the direction of the linear relationship between variable
  - **Correlation** - measures both the strength and direction of the linear relationship between two variables
  - correlation values are standardized whereas, covariance values are not
  - Covariance is affected by the change in scale, i.e. if all the value of one variable is multiplied by a constant and all the value of another variable are multiplied, by a similar or different constant, then the covariance is changed. As against this, correlation is not influenced by the change in scale.
  - Correlation is dimensionless, i.e. it is a unit-free measure of the relationship between variables. Unlike covariance, where the value is obtained by the product of the units of the two variables.
---

### Q. Different correlation coefficient's and their significance? Which one to use when?
  - **Pearson Correlation Coefficient** - The Pearson (product-moment) correlation coefficient is a measure of the linear relationship between two features
  - **Spearman Correlation Coefficient** - The Spearman correlation coefficient between two features is the Pearson correlation coefficient between their rank values. It’s calculated the same way as the Pearson correlation coefficient but takes into account their ranks instead of their values.
  - **Kendall Correlation Coefficient** - Let’s start again by considering two n-tuples, x and y. Each of the x-y pairs (x₁, y₁), (x₂, y₂), … is a single observation. A pair of observations (xᵢ, yᵢ) and (xⱼ, yⱼ), where i < j, will be one of three things:
	- concordant if either (xᵢ > xⱼ and yᵢ > yⱼ) or (xᵢ < xⱼ and yᵢ < yⱼ)
	- discordant if either (xᵢ < xⱼ and yᵢ > yⱼ) or (xᵢ > xⱼ and yᵢ < yⱼ)
	- neither if there’s a tie in x (xᵢ = xⱼ) or a tie in y (yᵢ = yⱼ)
	
The Kendall correlation coefficient compares the number of concordant and discordant pairs of data. This coefficient is based on the difference in the counts of concordant and discordant pairs relative to the number of x-y pairs. 

**Reference**
- [How to choose between Pearson and Spearman correlation?](https://stats.stackexchange.com/questions/8071/how-to-choose-between-pearson-and-spearman-correlation)
- [Clearly explained: Pearson V/S Spearman Correlation Coefficient](https://towardsdatascience.com/clearly-explained-pearson-v-s-spearman-correlation-coefficient-ada2f473b8)
- [Comparison of Pearson and Spearman correlation coefficients](https://www.analyticsvidhya.com/blog/2021/03/comparison-of-pearson-and-spearman-correlation-coefficients/#:~:text=Pearson%20correlation%3A%20Pearson%20correlation%20evaluates,rather%20than%20the%20raw%20data.)

---

### Q. How pearson correlation coefficent works for categorical data?
By doing one-hot encoding/encoding of categorical variables (then they will be converted to numeric form)
	
---

### Q. How to check correlation for Categorical features?

#### <ins>Correlation Between Continuous & Categorical Features</ins>
 - **Case 1:** When an Independent Variable Only Has Two Values - If a categorical variable only has two values (i.e. true/false), then we can convert it into a numeric datatype (0 and 1). Since it becomes a numeric variable, we can find out the correlation using the dataframe.corr() function.
 - **Case 2:** When Independent Variables Have More Than Two Values - Use ANNOVA

**Links:**
 - [Find Correlation Between Categorical and Continuous Variables](https://dzone.com/articles/correlation-between-categorical-and-continuous-var-1#:~:text=Point%20Biserial%20Correlation,corr()%20function.)
 - [Correlation Between Continuous & Categorical Variables](http://www.ce.memphis.edu/7012/L17_CategoricalVariableAssociation.pdf)
 - [How to quantify relationship between categorical and continuous variables](https://edvancer.in/DESCRIPTIVE+STATISTICS+FOR+DATA+SCIENCE-2)
 - [Correlation between a nominal (IV) and a continuous (DV) variable](https://stats.stackexchange.com/questions/119835/correlation-between-a-nominal-iv-and-a-continuous-dv-variable/124618#124618)
 - [How to measure the correlation between a numeric and a categorical variable](https://thinkingneuron.com/how-to-measure-the-correlation-between-a-numeric-and-a-categorical-variable-in-python/)
 - [The Search for Categorical Correlation](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)


#### <ins>Correlation b/w categorical variables</ins>
For checking correlation between 2 categorical vaiables use "Chi-Square test" or ' Crammer's V'

**Links:**
 - [How to get correlation between two categorical variable and a categorical variable and continuous variable?](https://datascience.stackexchange.com/questions/893/how-to-get-correlation-between-two-categorical-variable-and-a-categorical-variab)
 - [The Search for Categorical Correlation](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)

 - **One-Sample proportion Test** - For One categorical variable
 - **Chi-square Test** - For two categorical variables
 - **T-Test** - One continuous variable or two-continuous variables
 - **T-test** - one categorical(having 2 categories) and one continuous variable
 - **ANNOVA Test** - one or more categorical(having more than 2 categories) and one continuous variable

---

### Q. Difference between Z-test, T-test and F-test
  - [Statistical Tests — When to use Which ?](https://towardsdatascience.com/statistical-tests-when-to-use-which-704557554740)
  - [Hypothesis testing; z test, t-test. f-test](https://www.slideshare.net/shakehandwithlife/hypothesis-testing-z-test-ttest-ftest)
---

### Q. Techniques of Outlier Detection
 - Elliptic Envelope [(here)](https://towardsdatascience.com/two-outlier-detection-techniques-you-should-know-in-2021-1454bef89331)
 - IQR Based Detection [(here)](https://towardsdatascience.com/two-outlier-detection-techniques-you-should-know-in-2021-1454bef89331)
 - Using Mahalanobis Distance - [Multivariate Outlier Detection in Python](https://towardsdatascience.com/multivariate-outlier-detection-in-python-e946cfc843b3)

---

### Q. Missing Values Imputation
 - Do Nothing
 - Imputation Using (Mean/Median) Values
 - Imputation Using (Most Frequent) or (Zero/Constant) Values
 - Imputation Using k-NN
 - Imputation Using Multivariate Imputation by Chained Equation (MICE)
 - Imputation Using Deep Learning (Datawig)

---

### Q. When to use mean and median Imputation?
1. If a variable is normally distributed, the mean, median, and mode, are approximately the same. Therefore, replacing missing values by the mean and the median are equivalent. 
2. If the variable is skewed, the mean is biased by the values at the far end of the distribution. Therefore, the median is a better representation of the majority of the values in the variable.

---

### Q. How does SMOTE work? [(here)](https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5#:~:text=SMOTE%20works%20by%20utilizing%20a,randomly%20selected%20k%2Dnearest%20neighbor.)

---

### Q. Techniques for features or variable selection
 - **Univariate Selection** - Statistical tests can be used to select those features that have the strongest relationship with the output variable.  For example the **ANOVA F-value** method is appropriate for numerical inputs and categorical data, as we see in the Pima dataset. This can be used via the **f_classif()** function in  **SelectKBest** class of scikit-learn library library that can be used with a suite of different statistical tests to select a specific number of features.
 - **Recursive Feature Elimination** - The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
 - **Principal Component Analysis** - Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.
 - **Feature Importance** - Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.

---

### Q. Dimensionality Reduction algorithms
 - Principal Component Analysis(PCA)
 - Singular Value Decomposition(SVD)
 - Linear Discriminant Analysis(LDA)
 - Independent Component Analysis(ICA)
 - Multi dimensional scaling(MDS)
 - Isomap Embedding
 - Locally Linear Embedding
 - Modified Locally Linear Embedding
 - t-distributed stochastic neighbour embedding(t-SNE)

---

### Q. Is there any basis to decide the train and test split ratio?

---

### Q. Linear Regression parameters definition
 - **Standard Error** - It measures the variability in the estimate of the coefficients. A lower value of standard error is good but it is somewhat relative to the value of coefficients.
 - **t-value** - It is the ratio of the estimated coefficients to the standard deviation of the estimated coefficients. It measures whethr or not the coefficient for this variable is meaningful for the model.
 - **p-value** - p-value indicates the probability of an event/result occured randomly. Lower p-value indicates the result was unlikely to have occured by chance alone/randomly. We will define some significance level(alpha=0.05 or 0.01) to check for p-value
 
 p-value < 0.05, we will reject the null hypothesis(as we have less evidence or the chance to occur)
 
 p-value > 0.05 fail to reject the null hypothesis
 - **R-Squared** - It is a statistical measure of how close the data are to the fitted regression line. **Adjusted R-Squared** is a better metric tahn R-squared to assess how good model fits the data. R-squared always increases if additional variables are added into the model, even if they are not related to the dependent variable.

 Adjsuted R-squared, penalise R-squared for unnecessary addition of variables, if the variable added does not increase the accuracy, Adjusted R-square decreases.

---

### Q. Linear Regression assumptions
 - **Linear Relationship** - check by residual vs fitted value plots
 - **No auto-correlation** - Durbin-Watson Test, value lie between 0 and 4, DW=2 no auto-correlation
 - **No Multicollinearity** - calculate correlation and VIF
 - **Multivariate Normality** - check by plotting Histogram and Q-Q plot, Normality can be checked with Kolmogorov-Smirnov test(KS), when data is not normally distributive a non-linear transformation log(x), √x, x² might fix the issue
 - **Homoscedasticity** - Error terms(or residual) are equal across the regression line. Residual Vs fitted values plot (Funner shape pattern)

---

### Q. Can we use Linear Regression for classification problem?

---

### Q. Why do we take sum of square in Linear Regression?

---

### Q. What’s the Difference Between RMSE and RMSLE (Evaluation metrics for Linear Regression)? [(here)](https://medium.com/analytics-vidhya/root-mean-square-log-error-rmse-vs-rmlse-935c6cc1802a)
Using **RMSLE** as evaluation metrics have below characterstics-
 - Robustness to the effect of the outliers
 - Calculates Relative Error
 - Biased Penalty (RMSLE incurs a larger penalty for the underestimation of the Actual variable than the Overestimation)

---

### Q. Difference betweence correlation and collinearity
**Correlation** - How two values are moving with each other and their direction

**Collinearity** - When two variables are so highly correlated that they explain each other (to the point that you can predict the one variable with the other), then we have collinearity (or multicollinearity).

---

### Q. If two variables are correlated, How to decide which one to remove?

---

### Q. How does Variance Inflation Factor(VIF) Work?
Regress each of the independent varables w.r.t rest of the independent variables in the model and calculate the R2 for each. Using R2 we can calculate the VIF of each variable i.e. VIF=1/(1-R2). Higher R2 value of independent variable corresponds to the high correlation, means the variable need to be removed.

---

### Q. Effect of Multicollinearity
Moderate multicollinearity may not be problematic. However, severe multicollinearity is a problem because it can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable and difficult to interpret. Multicollinearity saps the statistical power of the analysis, can cause the coefficients to switch signs, and makes it more difficult to specify the correct model.

---

### Q. What are Eigenvalues and Eigenvectors? [(here)](https://medium.com/fintechexplained/what-are-eigenvalues-and-eigenvectors-a-must-know-concept-for-machine-learning-80d0fd330e47#:~:text=Eigenvectors%20and%20eigenvalues%20revolve%20around,to%20represent%20a%20large%20matrix.)

---

### Q. What is PCA?
Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

---

### Q. How does Principal Component Analysis(PCA) works? [(here)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
  - Standardization - standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.
  - Covariance Matrix computation - to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them. Because sometimes, variables are highly correlated in such a way that they contain redundant information. So, in order to identify these correlations, we compute the covariance matrix.
  - Compute the Eigenvectors and Eigenvalues of the covariance matrix to identify the principal components
  - Feature vector
  - Recast the data along the principal component axes
  - [How to Calculate Principal Component Analysis (PCA) from Scratch in Python](https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)

---

### Q. Mathematics of Eigenvalues and Eigen vectors
  - A- XI (Where A is the matrix, X Is lambda, I is identity matrix)
  - take the determinent of A-XI i.e. det(A-XI) = 0
  - Solve of lambda, which wich will give the Eigen values
  - Using Eigen values get the Eigen vector (having unit length)

---

### Q. How PCA take cares of multicollinearity**
As Principle components are orthogonal to each other which helps in to get rid of multicollineraity

---

### Q. Why the Principal components are othogonal to each other?
Two vectors are uncorrelated (or independent of each other) when they are perpendicular(i.e. a.b = abcosθ)

---

### Q. Difference between PCA and Random Forest for feature selection.

---

### Q. How can we overcome Overfitting in a Regression Model?**
  - Reduce the model complexity
  - Regularization
    - **Ridge Regression(L2 Regularization)**
      - It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
      - It reduces the model complexity by coefficient shrinkage.
    - **Lasso Regression(L1 Regularization)**
      - It is generally used when we have more number of features, because it automatically does feature selection.

---

### Q. How to prevent overfitting [(here)](https://elitedatascience.com/overfitting-in-machine-learning)
   - Cross-validation
   - Train with more data
   - Remove features
   - Early stopping
   - Regularization
   - Ensembling

---

### Q. How to explain gain and lift to business person?

---

### Q. How you will define Precision

---

### Q. How to handle class imbalance problem?
  - Get more data
  - Try different performance matrix
    - Confusion Matrix
    - Precision
    - Recall
    - F1-Score
    - Kappa
    - Area Under ROC curve
  - Data Resampling
    - Undersampling
    - Oversampling
  - Generate synthetic data
  - Use different algorithms for classification
  - Try Penalized models
    - penalized-SVM
    - penalized LDA
  - Try different techniques
    - Anomaly detection
    - Change detection

---

### Q. What are the shortcomings of ROC AUC curve?**
  - https://www.kaggle.com/lct14558/imbalanced-data-why-you-should-not-use-roc-curve
  - https://stats.stackexchange.com/questions/193138/roc-curve-drawbacks

---

### Q. Why **"log loss"** as evaluation metrics not **"Mean squared error"** for Logistic Regression? [(here)](https://towardsdatascience.com/why-not-mse-as-a-loss-function-for-logistic-regression-589816b5e03c)

---

### Q. When to use Logistic Regression vs SVM? or Differences between Logistic Regression and SVM**
  - Logistic loss diverges faster than hinge loss. So, in general, it will be more sensitive to outliers.
  - Logistic loss does not go to zero even if the point is classified sufficiently confidently. This might lead to minor degradation in accuracy.
  - SVM try to maximize the margin between the closest support vectors while LR the posterior class probability. Thus, SVM find a solution which is as fare as possible for the two categories while LR has not this property.
  - LR is more sensitive to outliers than SVM because the cost function of LR diverges faster than those of SVM.
  - Logistic Regression produces probabilistic values while SVM produces 1 or 0. So in a few words LR makes not absolute prediction and it does not assume data is enough to give a final decision. This maybe be good property when what we want is an estimation or we do not have high confidence into data.
  
  > **_Note:_**
  - SVM tries to find the widest possible separating margin, while Logistic Regression optimizes the log likelihood function, with probabilities modeled by the sigmoid function.
  - SVM extends by using kernel tricks, transforming datasets into rich features space, so that complex problems can be still dealt with in the same “linear” fashion in the lifted hyper space.

---

### Q. Logistic Regression Vs GLM 
Logistic Regression is the special case of GLM with  `distribution type=Bernoulli` and `LinkFunction=Logit`. Below are the various linear models we can run by changing the **distribution type** and **LinkFunction**

|Distribution Type    	  | LinkFunction | PredictFactor | ComponentModel     |
|-------------------------|--------------|---------------|--------------------|
|Normal                   | Identity     | Continuous    | Linear Regression  |
|Normal                   | Identity     | Categorical   | Anal. of Variance  |
|Normal                   | Identity     | Mixed         | Anal. of Covariance|
|Bernoulli                | Logit        | Mixed         | Logistic Regression|
|Poisson                  | Log          | Categorical   | Log-linear         |
|Poisson                  | Log          | Mixed         | Poisson Regression |
|Gamma(Positive ontinuous)| Log          | Mixed         | Gamma Regression   |

---

### Q. Working of multiclass classification

---

### Q. What is Gradient Descent
Gradient descent is used to minimize the cost function or any other function

---

### Q. What is the difference between Gradient Descent and Stochastic Gradient Descent? [(here)](https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent#:~:text=In%20Gradient%20Descent%20or%20Batch,of%20training%20data%20per%20epoch)

---

### Q. Working of Gradient Descent

---

### Q. Working of Gradient Boosting Machines(GBM)

#### <ins>Gradient Boost For Regression</ins>
1. We start with a leaf that is the average value of the variable we want to predict
2. Then we add tree based on the residuals, the difference between the observed values and the predicted values and we scale the tree's contribution to the final Prediction with a Learning Rate
3. Then we add another tree based on the new residuals and we keep on adding trees based on the errors made by the previous trees


#### <ins>Gradient Boost For Classification</ins>
1. In Gradient Boost for Classification, the initial Prediction for every individual is the log(odds). I like to think of thr log(odds) as the Logistic Regression equivalent of average
2. Convert log(odds) in to probability using Logistic Function (e^log(odds)/1+e^log(odds)) and this will be the initial prediction/probability for all the records
3. Then calculate the Residuals(to measure how bad the prediction is), difference between Observed and the predicted values, Residual = (Observed - Predicted)
4. Then we add tree based on the residuals and calculate the output values for the leaves and tranform in to probablities using (sum of all the Residuals in the leaf/sum of (previous probability*(1-previous probability)))
	- Calculate the log(odds) prediction = previous prediction + output value from the tree scaled by the learning rate
	- Then conversts the log(odds) prediction to probability
	- Do above 2 steps for all the records
	- Then repeat the steps 3 and 4 until we get the optimized predictions

> **_Note:_** we can limit the number of leaves allows in a tree

---

### Q. How does ExtraTreeRegressor different from Random Forest? [(here)](https://stats.stackexchange.com/questions/175523/difference-between-random-forest-and-extremely-randomized-trees)

|Features                                                  |Decision Tree   | Random Forest           |Extra Trees               |
|:---------------------------------------------------------|:---------------|:------------------------|:-------------------------|
|Number of trees                                           |1               |Many                     |Many                      |
|No. Of Features considered for split at each decision mode|All Features    |Random subset of Features|Random subset of Features |
|Bootstrapping(Drawing sample with replacement)            |Not Applied     |Yes                      |No                        |
|How split is made                                         |Best Split      |Best Split               |Random Split              |

**Refrences:**
 - [What is the difference between Extra Trees and Random Forest?](https://quantdare.com/what-is-the-difference-between-extra-trees-and-random-forest/)
 - [Extremely randomized trees](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)

---

### Q. How the tree built in Random Forest different from XGBoost?

---

### Q. What kind of separator tree-based algorithm have?

---

### Q. Distance Measures [(here)](https://machinelearningmastery.com/distance-measures-for-machine-learning/)
 - Hamming Distance
 - Euclidean Distance
 - Manhattan Distance (Taxicab or City Block)
 - Minkowski Distance
 - Mahalanobis Distance (For measuring the distance between a point and a distribution)

---

### Q. Algorithms for clustering
 - **K-Means** - For datasets having numerical variables
 - **Mini-Batch K-Means** - Mini-Batch K-Means is a modified version of k-means that makes updates to the cluster centroids using mini-batches of samples rather than the entire dataset, which can make it faster for large datasets, and perhaps more robust to statistical noise.
 - **Hierarchical clustering**
 - **K-Mode** - For datasets having categorical variables
 - **k-Prototype** - It combines the k-modes and k-means algorithms and is able to cluster mixed numerical and categorical variables
 - **PAM (Partitioning Around Medoids) or K-Medoids** - For datasets having both numerical and categorical variables
 - **BIRCH**
 - **DBSCAN**
 - **Mean Shift**
 - **OPTICS**

---

### Q. How to find optimal value of k (or number of clusters) in clustering?
 - Elbow method
 - Silhouette coefficient 

---

### Q. Evaluation metrics for measuring the effectiveness of clusters formed
 - Dunn Index
 - Silhouette analysis

---

### Q. How to compute the mean Silhouette Coefficient of all samples?
**Silhouette** refers to a method of interpretation and validation of consistency within clusters of data

The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from `−1 to +1`, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters. The silhouette can be calculated with any distance metric, such as the **Euclidean distance** or the **Manhattan distance.**

The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is **(b - a) / max(a, b)**. To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.

---

### Q. How will you check if the segmenting is good or whether you need to use different factors for segmenting(K-Means)?
Through Silhouette analysis

---

### Q. What are the benefits of Hierarchical Clustering over K-Means clustering? What are the disadvantages?
Hierarchical clustering generally produces better clusters, but is more computationallyintensive.

---

### Q. Will the result of K-Means vary in each iteration (means executed multiple times from starting)

---

### Q. How to check convergence in K-Means

---

### Q. How to calculate Gower’s Distance using Python [(here)](https://medium.com/analytics-vidhya/concept-of-gowers-distance-and-it-s-application-using-python-b08cf6139ac2)

---

### Q. How to perform clustering on large dataset?

---

### Q. How the recommendation system work if I don't like/dislike the any movies (in case of Netflix), just simply watch the movies there then, How the rating will be given (means the User vector is defined)?

---

### Q. Hyrid recommendation system

---

### Q. How to measure the performance of recommendation system?

---

### Q. ALS (alternating Least Squares) for Collaborative Filtering (algorithm for recommendation)
  - ALS recommender is a matrix factorization algorithm that uses Alternating Least Squares with Weighted-Lamda-Regularization (ALS-WR). It factors the user to item matrix A into the user-to-feature matrix U and the item-to-feature matrix M: It runs the ALS algorithm in a parallel fashion. The ALS algorithm should uncover the latent factors that explain the observed user to item ratings and tries to find optimal factor weights to minimize the least squares between predicted and actual ratings.
  - [How does Netflix recommend movies? Matrix Factorization](https://www.youtube.com/watch?v=ZspR5PZemcs)
  - [Prototyping a Recommender System Step by Step Part 2: Alternating Least Square (ALS) Matrix Factorization in Collaborative Filtering](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1)
  - [Explicit Matrix Factorization: ALS, SGD, and All That Jazz](https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea)
  - [ALS Implicit Collaborative Filtering](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe)
  - [Alternating Least Squares (ALS) Spark ML](https://www.elenacuoco.com/2016/12/22/alternating-least-squares-als-spark-ml/?cn-reloaded=1)

---

### Q. Alternating least squares algorithm (ALS)**
It holds one part of a model constant and doing OLS on the rest; then assuming the OLS coefficients and holding that part of the model constant to do OLS on the part of the model that was held constant the first time. The process is repeated until it converges. It's a way of breaking complex estimation or optimizations into linear pieces that can be used to iterate to an answer.

---

### Q. Hyperparameters to tune in Logistic Regression**
  - Logistic regression does not really have any critical hyperparameters to tune.
    - Sometimes, you can see useful differences in performance or convergence with different solvers (solver).
      - **solver** in [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
    - Regularization (penalty) can sometimes be helpful.
      - **penalty** in [‘none’, ‘l1’, ‘l2’, ‘elasticnet’]

---

### Q. Hyperparameters to tune in Random Forest**
  - **n_estimators** = number of trees in the foreset
  - **max_features** = max number of features considered for splitting a node
  - **max_depth** = max number of levels in each decision tree
  - **min_samples_split** = min number of data points placed in a node before the node is split
  - **min_samples_leaf** = min number of data points allowed in a leaf node
  - **bootstrap** = method for sampling data points (with or without replacement)

---

### Q. Hyperparameters to tune in XGBoost** [(here)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

---

### Q. How Parallelization works in XGBoost giving that boosting technique build models sequentially? [(here)](http://zhanpengfang.github.io/418home.html)

---

### Q. How to visualize the data pattern in dataset having large number of featues (Know this will help in type of algorithm to fit)?

---

### Q. Difference between Gridsearch and Random search

---

### Q. How to choose value of k in k-fold cross validation

---

### Q. All about When and How to do train_test_split and pre_processing
  - [3 Things You Need To Know Before You Train-Test Split](https://towardsdatascience.com/3-things-you-need-to-know-before-you-train-test-split-869dfabb7e50)
  - [How to Avoid Data Leakage When Performing Data Preparation](https://machinelearningmastery.com/data-preparation-without-data-leakage/)
	
---

### Q. Anomaly Detection Unsupervised Algorithms for Time-series data[(here)](https://towardsdatascience.com/unsupervised-anomaly-detection-on-time-series-9bcee10ab473)
1. **Probability Based Approaches**
	- Using Z-score
	- Quartiles-based
	- Elliptic Envelope
2. **Forecasting Based Approaches**
	- LSTM
	- ARIMA
	- Prophet
3. **Neural Network Based Approaches**
	- Autoencoder
	- Self Organizing Maps (SOM)
4. **Clustering Based Approaches**
	- k-means
	- Gaussian Mixture Model (GMM)
	- DBSCAN
5. **Proximity Based Approaches**
	- k-nearest neighbor(k-NN)
	- Local Outlier Factor (LOF)
6. **Tree Based Approaches**
	- Isolation Forest
	- Extended Isolation Forest
7. **Dimension Reduction Based Approaches**
	- Principal Component Analyses (PCA)

---

### Q. How to handle high cardinality categorical features in model building?

---

### Q. Ways of Encoding categorical variables

### 1. Classic Encoders

- Ordinal — convert string labels to integer values 1 through k. Ordinal.
- OneHot — one column for each value to compare vs. all other values. Nominal, ordinal.
- Binary — convert each integer to binary digits. Each binary digit gets one column. Some info loss but fewer dimensions. Ordinal.
- BaseN — Ordinal, Binary, or higher encoding. Nominal, ordinal. Doesn’t add much functionality. Probably avoid.
- Hashing — Like OneHot but fewer dimensions, some info loss due to collisions. Nominal, ordinal.
- Sum — Just like OneHot except one value is held constant and encoded as -1 across all columns.

### 2. Contrast Encoders

- Helmert (reverse) — The mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels.
- Backward Difference — the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level.
- Polynomial — orthogonal polynomial contrasts. The coefficients taken on by polynomial coding for k=4 levels are the linear, quadratic, and cubic trends in the categorical variable.

### 3. Bayesian Encoders

- Target — use the mean of the DV, must take steps to avoid overfitting/ response leakage. Nominal, ordinal. For classification tasks.
- LeaveOneOut — similar to target but avoids contamination. Nominal, ordinal. For classification tasks.
- WeightOfEvidence — added in v1.3. Not documented in the docs as of April 11, 2019. The method is explained in this post.
- James-Stein — forthcoming in v1.4. Described in the code here.
- M-estimator — forthcoming in v1.4. Described in the code here. Simplified target encoder.

---

### Q. Mistakes while performing A/B Testing and their Remedies

**1. Mistake novelty effect as real effect**\
Novelty effect is when customers engage with a new feature simply because it's new, but not because they like it. You might see the treatment gets more engagement than control in the beginning, but it's not the real effect.

**Remedy:** instead of analyzing all customers with the same cut-off for start time and end time, do a cohort analysis based on when customers get assigned to the treatment group, and see whether this effect wears off with time.

**2. Cannibalization**\
The treatment you test might have a positive impact for your experiment, but it might hurt other features on the website, and that's why it's call 'cannibalization'.

**Remedy:** use linear models to estimate interaction effect. You can create four cohorts and analyze them: the cohort that are in controls for both experiments, the cohort that in treatments for both experiments, and also isolate the cohorts that are only in one experiment. You need to make sure the cohorts have similar duration of exposure to compare.

But sometimes the effect is hard to measure, so before you design an experiment, work with product managers and ux researchers to understand the entire customer journey, top priority business metrics to optimize for, instead of just focusing on the performance of your experiment. Understand the ecosystem of where your feature lives in.

**3. "Cherry-pick" metrics for launch decisions**\
Sometimes decision makers really want to launch a feature, and they would pick whatever looks positive to support their decisions, which violates the principals of statistical testing.

**Remedy:** have no more than three core metrics, and have 1-2 secondary metrics to make decisions. The key is to decide on those metrics BEFORE you start experimentation, and stick to them. Don't move the goal posts just because you want to launch it. If you see something interesting, investigate it, treat it as a new assumption, and don't make your decision based on an unexpected change.


---
