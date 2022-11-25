# Gaming Anxiety Prediction ~ a regression and classification analysis using Python

<details>
  <summary>Table of Contents</summary>
  <ol>
    
* [About The Project](#about-the-project)
* [Data Sources](#data-sources)
* [Data Description](#data-description)
* [Data Mining Tasks](#data-mining-tasks)
* [Model Exploration and Model Selection](#model-exploration-and-model-selection)
* [Impact of the Project Outcomes](#impact-of-the-project-outcomes)
    
  </ol>
</details>

## About The Project

* Gaming has become a common aspect of many people’s lives. Video games are usually playedas an escape from reality, but 
addiction is a common problem associated with it. Gaming has been linked to anxiety disorders, negative attitudes, and low 
self-esteem especially when played in excess. Previous studies have exhibited that excessive engagement in games can lead 
to Internet Gaming Disorder (IGD) like GAD (General Anxiety Disorder) and SWL (Satisfaction withLife). Like any other compulsive
disorder, online game addiction can have severe negative short-term or long-term effects. To avoid these consequences, it is 
necessary to identify and provide necessary precautions based on the gamingfactors. 

* In this project, we intend to find the best model to determine connection between gaming habits, various socio-economic
factors and measures of anxiety, social phobia, life satisfaction and narcissism. We intend to forecast the influence of
online gaming on players' personal and professional lives.

## Data Sources

The dataset is obtained from Kaggle which provides usthe information of player’s gaming routine,
country, and anxiety disorderscoring.
Link: https://www.kaggle.com/kerneler/starter-online-gaming-anxiety-data-54e0ee82-0/data

## Data Description

This dataset consists of 55 columns of data collected as a part of a survey among gamers worldwide. The questionnaire 
asked questions that psychologists generally ask people who are prone to anxiety,social phobia, and lessto no life
satisfaction. The questionnaire consists of several set of questions as asked as a part of psychologicalstudy.

*Features:*

- Timestamp – Time at which the participant took the questionnaire after it being launched.
- GAD 1-7 - Response to GAD question. General Anxiety Disorder is calculated by assigning
scores of 0, 1, 2, and 3 to the response categories, respectively, of “not at all,” “several days,”
“more than half the days,” and “nearly every day.”
- GADE - Effect of gaming in work.
- SWL 1-5 - Response to SWL questions. Satisfaction With Life Scale is a short 5-item instrument
designed to measure global cognitive judgments of satisfaction with one’s life.
- Game - Name of the game they play.
- Platform - Mode of game playing (PC, Console, Mobile etc.)
- Hours - Number of hours in a week devoted to playing.
- Earnings - Earnings from the game (if any)
- Whyplay - Reason to play the game.
- League – League of the game
- Highest League – Highest League
- Streams – Number of online streaming of the game.
- SPIN 1 -17 - Response to SPIN questions. The Social Phobia Inventory (“SPIN”) is a 17-item
  self-rating for social anxiety disorder (or social phobia).
- Narcissism - Interest scale in the game. (1-5)
- Gender – Self identified gender of the gamer taking the questionnaire.
- Age – Self reported age of the gamer taking the questionnaire.
- Work - Work status of the gamer.
- Degree - Highest degree attained.
- Birthplace – Country of Birth
- Residence – Country of Residence
- Playstyle - Playstyle (Multiplayer, single player etc.)
- Accept - Accept terms and conditions (not necessary for any analysis)
- SWL_T - SWL Total Score
- SPIN_T - SPIN Total Score
- GAD_T - GAD Total Score

## Data Mining Tasks

### 1. Data Understanding:

Around 13500 participants, between the age 18 and 63 years completed the survey. Participants
resided in 109 different countries with most of the participants coming from the USA, Germany,
the UK, and Canada.

### 2. Data Cleaning and Pre-processing:

The dataset has 13464 rows and 55 columns. On examination, 29 columns were found to have null
values. The attribute ‘highestleague’ was dropped as it was unnecessary and had more than 75%
of null values. Attributes, ‘League’, ‘Reference’, ‘accept’, ‘Residence_ISO3’, ‘Birthplace_ISO3’
were also dropped as they were unnecessary for the data mining process.
The numerical attributes, ‘Hours’, ‘streams’, SPIN 1-17, ‘Narcissism’, ‘SPIN-T’ had less than
25% of null values. Hours, streams, Narcissism and SPIN 1-17 were imputed using the central
tendency measure, median as these values were skewed. The outliersin the columns, ‘Hours’ and
‘streams’ were removed as they were incoherent with overall data. The attribute, ‘SPIN-T’ is the
sum of all SPIN values of a gamer. This was computed by adding all SPIN values after imputation.
The categorical attributes, ‘GADE’ and ‘Work’ were also found to have less than 25% of null
values. As the columns are important for the task, the null values in the attribute ‘GADE’ were
imputed with the most common class (mode) and that under ‘Work’ were imputed by replacing
with the category “Unknown”.
As an outcome of the cleaning and pre-processing of the data, the dataset has a remaining of 50
columns and 13458 rows carrying out further data mining steps.

### 3. Data Exploration and Visualization:

* **Relation between the attributes**

To check the relation between each data we plotted correlation plot which explains the values near
to 1 are highly correlated. As we can see in the above plot, the SPIN and GAD have 0.45 which
says that people with high Social Phobia Inventory scores tend to have the General Anxiety
Disorder. 

* **Distribution of GAD and SPIN scores for all the age groups**

A scatter plot is plotted to see how male, female, and other genders are affected by GAD and SPIN.
From the below plots we can conclude that Males are affected from the anxiety disorder as the
frequency of Males are distributed over all the age groups. Female are not at all affected after around
the age 35. 

* **Hours spent based on Employment Status and Gender**

We plotted a Tree map to visualize the player’s employment status. Male players are dominating
and the players who go to College/ University are the ones who play online games more ( total of
around 138714 hours spent). Players who are unemployed stream the game for the total of 34775
hours. From the map we can conclude that players who are unemployed spend less time on online
games compared to the students and employed ones.

* **Impact of SPIN and SWL by hours**

We have plotted Box plots to show the effect of SPIN and SWL by number of hours the online
game is player. Here we have taken the X- axis by grouping 10 hours interval. From the first plot
we can see that Social Phobia Inventory score is highest for the players who stream for an average
of 60 hours, and it is less for the players who stream for 10 hours.
In the second plot, we have considered the satisfaction with life scores, which is highest for the
players who stream for 10 hours. 

* **Players from different educational degrees**

To find the player’s educational background, we plotted the bar chart. The players who just have
High school diploma tend to play more compared to any other degrees.

* **GAD score across the World**

To see which countries, have the most General Anxiety Disorder we have plotted a world map.
From the map we can see that the players of Russia and USA have the most Anxiety scores
compared to any other countries

## Model Exploration and Model Selection

### 1. Classification Models
A new column ‘GAD_cat’ is created using ‘GAD_T’ values classifying into different anxiety
levels. This column is used as the target variable for the classification models.

#### 1. Random Forest

The random forest algorithm consists of a group of decision trees, each of which is built up of a
data sample from a training set with replacement. The prediction will be determined differently
depending on the type of difficulty. Individual decision trees will be averaged in a regression
task, and a majority vote, i.e., the most common categorical variable, will produce the predicted
class in a classification task

Implementation:
• The GAD class i.e., anxiety level was predicted using Random Forest Classifier.
• To optimize the model, GridSearchcv was used and the best estimator used it
RandomForestClassifier(max_depth=5, n_estimators=24, random_state=0).
• The accuracy of the model is 62.12%

#### 2. Decision Tree Classifier

Decision Tree is a Supervised Machine Learning Algorithm that makes judgments based on a
set of rules, similar to how people do. The technique attempts to entirely separate the dataset so
that all leaf nodes, or those that do not further split the data, belong to a single class.

Implementation:
• Gridsearchcv is used to find the best estimators for improving accuracy,
DecisionTreeClassifier(max_leaf_nodes=26, random_state=42).
• The accuracy of this model is 62.001%

#### 3. Logistic Regression

Logistic regression is a classification system for determining the likelihood of an event's success
or failure. When the dependent variable is binary in nature (0/1, True/False, Yes/No), it is
employed. It aids in the classification of data into discrete classes by examining the link between
a set of labelled data. It takes the given dataset and learns a linear relationship before adding
non-linearity in the form of the Sigmoid function.

Implementation:
• Gridsearchcv is used to find the best estimators for improving accuracy,
LogisticRegression(C=10, solver='newton-cg').
• The accuracy of this model is 62.24%

#### 4. Ridge Classification

The Ridge Classifier, based on Ridge regression method, converts the label data into [-1, 1] and
solves the problem with regression method. The highest value in prediction is accepted as a
target class and for multiclass data multi-output regression is applied.

Implementation:
• Gridsearchcv is used to find the best estimators for improving accuracy,
RidgeClassifier(alpha=0.1).
• The accuracy of this model is 62.09%

#### 5. Support Vector Machine

The support vector machine algorithm's goal is to find a hyperplane in an N-dimensional space
(N — the number of characteristics) that categorizes the data points clearly. Our goal is to
discover a plane with the greatest margin, or the greatest distance between data points from
different classes.

Implementation:
• Gridsearchcv is used to find the best estimators for improving accuracy, SVC(C=1000,
gamma=0.01)
• The accuracy of this model is 61.85%

#### 6. Stochastic Gradient Boosting

GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of
arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the
negative gradient of the binomial or multinomial deviance loss function.

Implementation:
• Gridsearchcv is used to find the best estimators for improving accuracy,
GradientBoostingClassifier(learning_rate=1, max_depth=1, n_estimators=250).
• The accuracy of this model is 62.21%

### 2. Regression Models

Using ‘GAD_T’ as the target variable to predict the anxiety level of gamers.

#### 7. Linear Regression

Linear regression is a statistical classification model used to estimate the probability of an event
occurring having been given some previous data related to, which can be considered as predictor
data. It is based on Sigmoid function where it learns a linear relationship from the data and then
introduces a non- linearity. 

Implementation:
• For linear regression, we used the parameter GAD_T to find the output score which is R
squared value.
14
• The R-squared measure is based on the residuals which is the differences between what the
model predicts for each data point and the actual value of each data point which we found
out R2 value is 46.25% the RMSE value is 0.34. 

#### 8. Lasso Regression

Lasso regression employs shrinkage and is a form of linear regression. Data values are shrunk
towards a central point, such as the mean, in shrinkage. Simple, sparse models are encouraged
by the lasso approach (i.e. models with fewer parameters).

Implementation:
• The R2 score is 46.04% and the RMSE value is 3.446.

## Impact of the Project Outcomes

• The aim of the project was to predict the anxiety level of a gamer given a variety of factors.
By providing details of the gamer, we can find the anxiety level of a gamer and tackle
consequences related to it accordingly.

• The SPIN, SWL, Narcissism values have a major impact on the anxiety level. The higher the
SPIN level and Narcissism and lower the SWL level, greater is the anxiety. Men are affected
more than women with gaming anxiety.
