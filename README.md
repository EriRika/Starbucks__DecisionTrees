# Starbucks__DecisionTrees
Capstone project of Udacity course. Got event data of an experiment. Main task was to restructure the data and derive success or fail drivers

## Table of contents

- [Data used](#data-used)
- [Libraries used](#libraries-used)
- [Business Questions](#business_questions)
- [Content of Notebook](#content_of_notebook)
- [Summary](#summary)

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

## Business Questions
### Definition
- An offer is **active**, if it has been viewed by the customer and if it is still valid.
- An offer is **passive**, if it has been viewed by the customer and if it is still valid.
- Same holds true for transactions
- A transaction is **independent**, if it was done during a period, where no offer is valid
- If a transaction happens during a period, where two offers are valid, I allocate a share of the transaction to each offer
- I use reward and cost in the same way as rewards are costs for starbucks

### Questions

In this analysis I tried to answer 4 business questions
- Which offer drives most revenue? (Passive and active)
- Which offer costs most? (Passive rewards and active rewards)
- Which persons have the highest passive rewards? --> These persons would by anyway. We should identify them and save the costs
- Can I use machine learning to identify passive rewards drivers?

## Content of Notebook
1. Introduction
2. Dataset
3. Data Preparation basic
   1. portfolio
   2. profile
   3. transcript
4. Data understanding basic
   1. barplots of offers per person
   2. barplot of unique offers per person
   3. Offers per timeslot
5. Define cleaning function to get a temporary dataframe
   1. Get temporary dataframe before grouping
      a. Perform health checks to make sure that total sums stay the same after wrangling     
   2. Create two dataframes for machine learning

6. ETL pipeline
7. ML Pipeline  
8. ML insights
 

## Summary
- How expensive is a night in Seattle?
<br />Prices are around 100$, but there are many outliers
![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/Price%20boxplot.png)
![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/Price%20histogram.png)

- When should I visit Seattle?
 <br />Seattle is cheapest between October and May. On top od that one can save in average 8$, if you do not stay during the weekend
![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/Average%20Price%20per%20weekday.png)
![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/Average%20Price%20over%20time.png)

- Can I find great places without paying too much money?
 <br />Yes, looking at the rating, we can see that there is no correlation between a very good rating and the price
 ![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/scatter_review_scores_rating.png)

- What can I learn about the different districts and their vibes?
 ![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/Diverse%20District.PNG)
 <br />By extracing the top 10 adjectives and nouns per district, I was able to describe them

- Can I predict the price per listing?
 <br />I reached an R2 of 0.62, which is acceptable, but not particularly good. The model underpredicts especially very expensive listings. One reason could be that there is a higher order dependency of the price and some features.
 ![alt text](https://github.com/EriRika/airbnb_prices/blob/master/pictures/prediction_Ridge.png)

   
[Medium Post - What Secrets does Airbnb data tell me about Seattle?](https://erikagintautas.medium.com/what-secrets-does-airbnb-data-tell-me-about-seattle-49fba69eb362)


