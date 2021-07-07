# Starbucks__DecisionTrees
Capstone project of Udacity course. I received event data of an experiment with the accoring campaign informatione and demographic data. The main task was to restructure the data and derive success or fail drivers.

## Table of contents

- [Data used](#data-used)
- [Libraries used](#libraries-used)
- [Project Definition](#project-definition)
- [Analysis](#analysis)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)

## Data used
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record



## Libraries used

- matplotlib
- seaborn
- pandas
- numpy
- SQLAlchemy
- sklearn


## Project Definition
### Project Overview
This project tackles a typical marketing and business question. Which customers should I send campaigns to and can discount-driven campaigns be harmful for the company?

*"Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. "*- Udacity

It is important to keep in mind  that the data is simulated and hence not all of the findings refreöct real customer behavior.
The data can be found on my github profile. The project contains three files.

### Problem Statement
The biggest part of this project was to understand and to prepare the data. The data was event-driven data, which means that events can overlap. On top of that it was important to understand that offers can be viewed by the customer and rewards can be gained afterwards when a certain spend threshold was reached. But this can also happen, if the customer never viewed the offer and hence did not know about the offer.

### Definitions
- An offer is active, if it has been viewed by the customer and if it is still valid.
- An offer is passive, if it has not been viewed by the customer and if it is still valid.
- Same holds true for transactions
- A transaction is independent, if it was done during a period, where no offer is valid
- If a transaction happens during a period, where several offers are valid, I allocate a share of the transaction to each offer

I use reward and cost in the same way as rewards are costs for starbucks. 
Initially I considered to perform an A/B test, but I had no holdout group nor different time periods to compare against. Hence I decided to focus on minimizing rewards payed to customers, who never viewed the offer - minimize passive rewards.

### Questions
- Can I figure out which customers had high passive rewards (they did not intend to gain it)?
- Can I predict the amount of passive rewards?
- Can I identify passive rewards drivers?

### Metrics
I will perform regression models in order to predict the passive reward. 
For this project my final metric is the R-2 Score. It represents the proportion of variance (of y) that has been explained by the independent variables in the model.
    
-![R-2 Score](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/R2%20formula.PNG "R-2 Score")

## Analysis
### Data exploration and visualization
Datasets and Inputs
The project contains three datasets.

#### portfolio
There are in total 10 offers. All of the offers are unique in terms of their characteristic combination. This also means that I can either use the offer_id as a feature, or the combination of reward, duration and difficulty.

-![portfolio](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/portfolio.PNG "Portfolio")

#### profile
There are in total 17000 customers in the dataset. I will have to perform some data cleaning later as there is one group of customers, which appears to have an age of 118. This group has missing values in income and gender. I assume that this is due to some data privacy restrictions.

-![Profile](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/profile.PNG "Profile")

The age group 118 is by far the biggest age group and represents 13% of the customer base.

-![Histogram of ages](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/histogram_ages.png "Histogram of ages")
   
When filtering for the age=118 it turns out that all gender and income values are missing.

-![Age is 118](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/age_118.PNG "Age is 118")

When filtering for the age!=118 it turns out none of the values are missing.

-![Age is not 118](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/age_Not_118.PNG "Age is 118")

I will have to transform categorical variables into dummy variables, hence it is important to look at the unique values per category.

-![unique values by column](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/unique_by_col.PNG "unique values by column")



I created buckets of 0.1 quantiles for the age to perform some plots. There are more males than females in the dataset. Interestingly the gap between males and females decreases with the age of the customer

-![Distribution by Age and gender](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/customer_by_gender_and_gender.png "Distribution by Age and gender")

In this dataset there are more females with a very high income. Low to medium income has more male customers.

-![Histogram of income by gender](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/hist_by_gender.png "Hist by gender")


The income grows with increasing age.

-![Income boxplot by age bucket](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/income_by_age_bucket.png "Income boxplot by age bucket")
   
#### Transcripts
The transcript file contains 306534 events. It has no missing values.

-![Transcript data](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/transcript.PNG "Transcript data")

Not every person received the same amount of offers. There are 6 persons without an offer. Other than that everyone received at least one offer.

-![Offers per person](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/offers_per_person.png "Offers per person")

The offers were sent randomly, such that each offer was sent at approx. the same frequency per time. This means that it does not makes sense to look at "offer paths". 
Initially I thought that it might be interesting to look at the order at which a customer received offers. But there are 13727 unique offer paths for 17000 people.
It is also intersting to understand that there are only 6 distinct timeslots when an offer was sent. There are almost always 7 days between two offers, except of between slot 21 and 24. Since the campaigns have durations of up to 10 days I will have overlapping active offers.
   -![Frequency of offers per time slot](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/offers_per_time.png "Frequency of offers per time slot")
   


## Methodology
In order to create a final training dataframe I had to first merge all three datasets and create features and a temporary dataframe. 

### Data Preprocessing - transcripts
The transcripts data is event-driven. As seen in image 7 it has to be transformed first. I created 3 additional columns by extracting information from the event column and joined the data with portfolio to extract offer information. This was performed by function merge_trans_portfolio in helper_functions
- offer_id
- amount
- reward_from_value

   -![Transcript after first transformation](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/transcript%20after%20first%20transformation.PNG "Transcript after first transformation")

There was one big pitfall where I lost quite some time. Initially I assumed that each transaction belongs to exactly one offer_id. I realized that only three new columns could not catch all information. For each offer_id I needed a separate column to account for simultaneous offers. Hence I had to restructure. The restructurng function clean_data_per_offer can be found here helper_functions.

#### Restructuring function - For each offer id I create the following set of columns
- **id_start_time** - shows time of offer received from the moment it was received. Consecutive rows have the same number. Once an offer is completed I set this to 500. If the same - offer is sent a second time this shows the new start_time
- **id_duration** - As above logic but for the duration
- **id_age** - Age of the offer = time - id_start_time → Important in order to know if an offer is still valid
- **id_is_valid** - 1 where age <= duration, 0 else
- **id_viewed_time** - shows time of offer viewed from the moment it was viewed. coonsecutive logic is as in id_start_time. This is important to define, if a transaction and a - reward was active or passive
- **id_is_active** - id_viewed_time - id_start_time <= id_duration
- **id_active_transaction** - Amount of transaction, if transaction was active
- **id_passive_transaction** - Amount of transaction, if transaction was passive , but valid!
- **social** - 1 if offer was active and came through Social Media
- **email** - see above
- **mobile** - see above
- **web** - see above
- **reward_0, reward_2, reward_3, reward_5, reward_10** -flag 1 if offer has that reward
- **difficulty_0, difficulty_2, difficulty_3, difficulty_5**, difficulty_10- flag 1 if offer has that reward
- **reward_help_column** - reward of completed offer but backfilled to the previous rows
- **id_active_reward** - reward_help_column x id_is_active
- **id_passive_reward** - reward_help_column x (~id_is_active)

**Additional columns (not offer specific)**
- **active_offers_simul** - Sum of active offers at the same time
- **valid_offers_simul** - Sum of valid offers at the same time
- **independent_transaction** - value of transaction, which was done during a time, where no offer was running

My best friend in this exercise was the groupby with a lambda function to forward fill missing values. 
For every campaign (offer_id) I created separate columns like the start_time of the offer. The generated columns contained many NaN values.
   -![Transformed transcript data](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/transformed%20transcript%20data.PNG "Transformed transcript data") 


In the above example I want all further rows to be 0 until an offer is completed (or not valid any more).
Using groupby with ffill did the trick

   -![FFill](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/ffill.PNG "ffill") 
The ffill method fills all NaN values in all consecutive rows until it finds the next non-Nan value. I made use of this behaviour by creating blockers. Whenever an event was "offer completed" I filled in a random number (in my case 500)

   -![Blockers](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/blockers.PNG "blockers") 

### Data Preprocessing - Feature selection and dummy creation

Categorical features in the dataset
channels: email, web, mobile, social,
offer type: bogo, discount, informational
- **reward**: 0, 2, 3, 5,10,
- **difficulty**: 0, 5, 7, 10, 20
- **gender**: F, M, O, nan

We have to be careful with features like age and income though. E.g. a decision tree could only split age by < or ≥ on certain threshold, which implieas a rank. This is ok, as long as we know that e.g. the target variable tends to decrease or increase after a certain threshold. Another problem with features like age is that they are discrete with many values. If most of your features are categorical you will have many binary columns, while age, income and membership will be discrete.

#### Numerical features in the dataset
- **age**
- **income**
- **membership** (date int)

In my experience decision trees tend to overvalue such features and your most important features might become age and income. Therefore, I created age, income and membership buckets in order to treat them like categorical features.
I used pandas qcut to create buckets. It cuts an array based on quantiles.
I could have also used pandas.cut function, but I though that I catch more particularities of the data by using quantiles. Especially when it comes to outliers.
   -![Pandas qcut](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/qcut.PNG "pd.qcut") 


**My final list of features looks as follows**
- **Offer_was** _received_by_customer_flag: 'id_received_' +offer_id,
- **Offer_start_time**: 'id_start_time_' +offer_id
- **offers_received_by_channel**: 'email_count', 'web_count', 'mobile_count', 'social_count',
- **offers_received_by_type**: 'bogo_count', 'discount_count', 'informational_count'
- **offers_received_by_reward**: 'reward_0_count', 'reward_2_count', 'reward_3_count', 'reward_5_count', 'reward_10_count',
- **offers_received_by_difficulty**: 'difficulty_0_count','difficulty_5_count','difficulty_7_count','difficulty_10_count','difficulty_20_count'
- **customer_gender**: 'gender_F', 'gender_M',
 'gender_O', 'gender_nan'
- **customer_age**: 'age_bucket_0', 'age_bucket_1',
 'age_bucket_2', 'age_bucket_3', 'age_bucket_4', 'age_bucket_5',
 'age_bucket_6', 'age_bucket_7', 'age_bucket_8', 'age_bucket_9',
 'age_bucket_118'
- **customer_income**: 'income_bucket_(29999.999, 37000.0]',
 'income_bucket_(37000.0, 45000.0]', 'income_bucket_(45000.0, 52000.0]',
 'income_bucket_(52000.0, 57000.0]', 'income_bucket_(57000.0, 64000.0]',
 'income_bucket_(64000.0, 70000.0]', 'income_bucket_(70000.0, 75000.0]',
 'income_bucket_(75000.0, 85000.0]', 'income_bucket_(85000.0, 96000.0]',
 'income_bucket_(96000.0, 120000.0]', 'income_bucket_nan'
- **customer_membership_age**: 'membership_start_year_2013', 'membership_start_year_2014',
 'membership_start_year_2015', 'membership_start_year_2016',
 'membership_start_year_2017', 'membership_start_year_2018'

**My target variable is** 
- passive_reward_amount - Sum of valid rewards, which were reached by passive transactions

### Data Preprocessing - Final Dataframe
There were two final dataframe structures, which I considered for my training.

**One row per customer**, which contains information about which offer was sent, which was viewed, how many offers were running at the same time. Which offer was activated at what time and the person's demographic information.
- **Advantage**: This way the algorithm can learn about the impact of the combinationa and amount of orders sent during the 30 day period.
- **Disadvantage**: If an offer was sent more than once to the same person, I am loosing information about when and in which order it was sent, beacuse I have to take a min or max time-sent

**One row per customer and offer_id**, which contains almost the same information, but is much more sparse with more rows.
- **Advantage**: Algorithm can predict each offer per person separately
- **Disadvantage**: I am loosing information about what combination of offers was sent to the customer.

The basic difference in creating those dataframes was the groupby (either grouped by person, or by person and offer id). You can find the functions here helper_functions

### Implementation
Reminder: I was going to test two different input dataframes and to predict passive rewards by using DecisionTreeRegressor and RandomForestRegressor
I stored all dataframes the temporary file, the One-row-per-customer-and-per-offer and the One-row-per-customer as sqlite databases. The temp file is not necessary to run the prediction, but it was convenient to store it because it takes approx. 15 minutes to create it.

#### Creating a simple machine learning pipeline
After preparing the data and storing it in a sqlite database I wrote a loading function.

   -![load_data](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/load_data.PNG "load_data")

I didn't know which settings where the best for the methods, therefore I tried several combinations with GridSearchCV. The nice thing about it is that it even performs a cross-validation splitting strategy when looking for the best parameters.
Once it's done, you can look at the best parameters with the best_params_ method.

   -![Build Model](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/build_model.PNG "build_model")
   
####GridSearchCV
Using GridSearch allowed me to test several hyper parameters. I ran the first prediction on the One row per customer dataframe with the following hyper parameters
- criterion: [‘mse’, ‘friedman_mse’]
- max_depth: [3, 5, 10],
- max_features: [None, 20, 10]
- min_samples_leaf: [680, 1000, 100]

The best results were achieved by
- criterion: mse
- max_depth: 10
- max_features: None
- min_samples_leaf: 100

I ran the first prediction on the One row per customer dataframe and I was pretty disappointed.
- DecisionTreeRegressor - R2 0.149
- RandomForestRegressor - R2 0.178

Running the prediction on the One row per customer and per offer dataframe was even more disappointing
- DecisionTreeRegressor - R2 0.129
- RandomForestRegressor - R2 0.139

## Refinement
From that point on I continued working only with the One row per customer dataframe. I realized that predicting the total amount of passive rewards did not work well.

**Create or derive new features**
I figured that maybe I have to first classify if a customer gets a passive reward at all. I ran a classification (RandomForestClassifier) on the target variable has_passive reward first and used that as an input
- RandomForestRegressor with has_passive Classification - R2 0.131

Since this did not generate the desired result I started to doubt my input features. Especially the reward feature, where I removed the order by creating dummies. I decided to include a new feature, which sums up the total possible rewards per customer during the 30-day test phase
- RandomForestRegressor with total_award_possible- R2 0.174

This did not improve my model - I decided to include a feature which contains the average difficulty level per customer.
- RandomForestRegressor with total_award_possible & average_difficulty - R2 0.171

**Tune hyperparameters**
Based on the results from the first run I knew where to try further improvements. I retrained RandomForestRegressor with new parameters, while a included the winners of the first run. The final R2 was 0.186, which was the best.
- criterion: mse
- max_depth: [10, 15, 20]
- max_features: None
- min_samples_leaf: [100, 50]
- n_estimators: [100, 1000]
- bootstrap: [True, False]


**Use a combined model**
One last approach I tried was a mix between machine learning and a heuristic model.
predict if a customer has a passive reward with a classifier
multiply with the average reward possible

- I received an R2 of -0.003 on the test set.

# Results
## Model Evaluation and Validation
I used sklearns r2_score to evaluate my model. I trained it on 80% of the data and validated it on the remaining 20%.

   -![train_evaluate](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/train_evaluate.PNG "train_evaluate") 

In the end the best model was the simplest model without any additional features or adaptions
   -![R2 by models](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/R2_by_model.PNG "R2 by models") 

### Feature importances
When looking at the top 10 feature importances of the best performing model one can see that the most important features are the offer ids and not the offer_type, channel, difficulty, reward or duration. This makes sense since each offer id represents a unique combination of those and hence contains more information.
   -![Feature Importance](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/Feature%20Importances.PNG "Feature Importance")

## Justification
I must admit that I struggled to find a model, which was able to predict the passive rewards per customer at a satisfying level. The highest R2 score, which I reached was 0.178 by using RandomForestRegressor. 
RandomForest performs better than a simple DecisionTree because it is designed to be better than a simple decision tree. It fits a number of decision trees on sub-samples and uses averaging to improve the predictive accuracy.
The fact that I couldn't find or create any additional features to improve the model could mean that the existing features contained all the necessary information and that the new features did not create new value for the algorithm.

Using GridSearch has two significant advantages
- It tells me the best combination of parameters by running through all possible combinations
- It performs k-fold cross-validation

The combined heuristic model does not work well, because it multiplies two predictions, which increases the error. First it classifies and then it multiplies with an average value.

# Conclusion
## Reflection
In this project I analyzed data from the Starbucks rewards app. I looked at distributions of demographic data as well as the structure of the experiment. The main dataset was event-based and I had to transform it into a dataframe, which I can use as input for machine learning. I defined a target variable and ran several regression models and evaluations.

I found it particularly difficult to transform the event-based data into the right format. As described above it wasn't even 100% percent clear to me which format would be best suited for the training and that's why I initially tested with two dataframes. In addition it was very important to implement sense-checks inbetween. I realized in the middle of the project that the passive and active rewards of my final dataframe did not sum up to the original awards, which were paid. The same happened with the transactions. Adding those test was crucial.

Even though RandomForest performed better in my tests I took a look at the tree structure of a DecisionTree to see, if I can derive somebusiness rules from that. The most obvious customer group in general was the high income group. These customers tend to purchase anyway and hence 
Sklearn provides a class tree, which has a plot_tree method.

   -![Plot Tree](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/plot_tree.PNG "Plot tree") 

**Class 1 High passive reward:**
- Offer is not reward_5 offer
- Offer is reward_10 offer
- Income bucket is known
- Income is > = (96000–12000)

   -![Decision Tree](https://github.com/EriRika/Starbucks__DecisionTrees/blob/master/images/RegressionPersonOffer.png "Decision Tree")

I hoped to find a good regression model. This would be very useful in saving marketing spend for the company - and more satisfying for myself :) .

I am not sure how much the fact that the data is simulated affected my outcome. Maybe I would have been able to find better regression models with a real-world dataset. 

I learned some useful data-wrangling techniques here, but I wish I had picked a project with less data cleaning. I would have enjoyed to code an algorithm from scratch.

## Improvement
Finding a good prediction turned out to be pretty difficult. 

One could cluster the customers based on their demographic data. For each cluster we could calculate the average passive and active rewards per cluster and use that as the prediction. Or we can use these clusters as input variables.

Other than that I would like to improve my coding structure by using classes and creating own libraries.
Use my notebooks to see the code!

   
[Medium Post - How Decision trees can help to find success drivers?](https://erikagintautas.medium.com/how-decision-trees-can-help-to-find-success-drivers-17edce59e1be)


