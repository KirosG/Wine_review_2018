# Wine_review_NLP_Application

Can machine learning algorithms help Wine tasters?

# Can machine learning algorithms help Wine tasters?

![](https://cdn-images-1.medium.com/max/800/1*2ZD56U8AcReDPDtAk76-Ew.png)

wine and points

One of my friend studied did his masters thesis about wine. Now he is one of the best wine tasters as well as i can say a wine specialist. one day we came to the point that, not all wine tasters are perfect and right. Therefore, he asked me to proof my hypothesis using data. Then i collected data from the internet via web-scraping and from the[University of California Irvine’s Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php).

![](https://cdn-images-1.medium.com/max/800/1*E6L0eQpCgVbL1kEMwV9LDg.png)

The[UCI Machine Learning repository](http://archive.ics.uci.edu/ml/machine-learning-databases/wine/)has two sets of[wine data](https://archive.ics.uci.edu/ml/datasets/wine+quality). One dataset contains information on red wines, and the other for white wines. i am not interested on the type of wines rather i am much interested on the people who taste wine across the globe as their main profession. Then once i downloaded the necessary inputs , its always important to develop or include the technical definition of the features description for the metadata. Based on my sources i would like to give the data dictionary as follows :-

![](https://cdn-images-1.medium.com/max/800/1*TqeQOt2BYDl1IuJDc8R6Nw.png)

Data dictionary

#### Wine variety and tasting dillemas

![](https://cdn-images-1.medium.com/max/800/1*aXuZnnqdU2F0XV2huZpC1g.jpeg)

common win types

Imagine your self, its Saturday afternoon ,you’re seated at an ivory table with your loved ones in an Ikea-furnished room or since i live in DC, lets think you are seated in Barcelona Wine Bar 14th Street. A mustached old man in a tailcoat with big hat, like the Texas cowboys enters, and gently sets two different unmarked bottles of wine down on the table along with a crystal tasting glass in front of you and your loved ones. He informs you in his thick local accent called it your favourite classic accent, that one bottle is $15, and the other $500, and politely challenges you to taste them to identify the more expensive of the two.what do you do?i am not sure, since i am bad in wine tasting, i would not identify it, but the nerd friend i have who is working in the wine industry will only take him probably 10 seconds to tell me the not only the name of the wine but also when it was made and which country it was produced in addition to the ingredients they use. who ever you reading this, i bet you to try and check what percentage of the time you can identify the most expensive wines? if you are not good, this[study](https://www.theguardian.com/science/2011/apr/14/expensive-wine-cheap-plonk-taste)conducted by psychologists at Hertfordshire University, 578 people were asked to do the exact same exercise across a broad selection of both red and white wines. On average, each taster correctly identified the wine’s price category just 47% of the time for reds, and 53% of the time for whites. In other words, the probability of distinguishing correctly was akin to a coin flip!

**While most of us would like to think we can tell the difference, in reality, the average wine consumer struggles to differentiate wine based on price point.**

Side note: Experienced wine drinkers have shown an ability to buck this trend, so don’t lose hope in your wine drinking future. They simply know the characteristics to sniff out that are associated with pricier, high-end wines.

Let’s hold the discussion and see the wine review datasets

The wine review dataset contains 129,970 wine reviews which include country, wine description, points, price, type of wine, reviewer’s name, etc.. The points (ranged from 0 to 100) represents the taste of wine. Higher point indicates better taste. The aim is to build a predictive model and find out the better taste wine in each categories such as dry tasted wine, fruity tasted wine etc.. Since so many variety of wine available in this dataset, its important to only focus with top ones that would be common to analyse and compare. Therefore, in order to reduce the data i used boolean method of filtering. during the process of data integrity and quality check as well as my initial exploratory data analysis i choose only to include wines with more than 1000 reviews only ignoring the types of wines with less than 1000 reviews and the basic reason is to compare across different countries and to see the type of wines that have better customer review and feedbacks as the dataset contains many countries like US,Portugal,France,Australia, Germany, Argentina, Chile, Austria, New-Zealand, Israel and others.

Can we predict the price of a bottle of wine from tasting or just looking the descriptions as well as variety??

Once i filter them out my data looks something similar to the following.

![](https://cdn-images-1.medium.com/max/800/1*Q-XFzJZVAUqo9DjPbPH-rA.png)

wine tasters

![](https://cdn-images-1.medium.com/max/800/1*b-ENRBzz2xBqGnwnuxC9qw.png)

Distribution of wines by customer rating

predicting the price of a single bottle of wine based on the variety and description is possible using different data science and machine learning tools. We can’t definitively say that wines with the word “cherry” in the description are more expensive, or that wines with “vanilla” are cheaper**.**As you can wee above there is some variability to some extent in almost all the top 24 types of wines no matter the customer rating or the tasters experience.

![](https://cdn-images-1.medium.com/max/800/0*b97gDLjmXHrixWHZ)

can we predict the price of wine by tasting ?

My goal here is to see if there is any kind of association or relationship between the tasters experience and as its shown below the most expensive wines have some better scores in review as compared to the others with less review or points.

![](https://cdn-images-1.medium.com/max/800/1*3sY5M3ttDEqJcz8mk-sqCg.png)

review versus price

Interestingly i run pearson correlation between price and point and found out that, there’s not significant correlation between the cost of wine and its rating. The result i generated was 0.477 which is moderate relationship and i was expecting some what a relationship between 0.5 ~ 1.0 to say there is strong relationship.

![](https://cdn-images-1.medium.com/max/800/1*NfucRDykbwAZzpETWdibfg.png)

price of wine and rating

As shown above there is an average $1.24 increase for every one point increase in rating according to my OLS results. very interesting point is how much focus and attention do we give to the wine tasting time and play. To me the imagination and sense during wine time as compared to other type of drinks makes much more sense, to get satisfaction, give more attention to your preferred type of wine to get the maximum utility or pleasure. The colour of the wine is the think while enjoying the discussion with loved ones to give attention not the smell .

![](https://cdn-images-1.medium.com/max/800/1*qVI9XPS_Cn3F6NtecD8_Ug.jpeg)

taste of wines

As i said it above, i enjoy drinking wine but i am very bad at tasting. Thanks to the nerd, wine ninja friend, i do not care, what type of wine i am going to order, he will be the one to choose.

![](https://cdn-images-1.medium.com/max/800/1*2Dbgc8V8LywYm5-nXwijUQ.png)

distribution of my dataset when i classify wine

![](https://cdn-images-1.medium.com/max/800/1*CG5C4rT7vJgjS5AyliuudQ.png)

The mean distribution of better and worse wine

The recommendation i will give you is that if you are drinking wine and if your wine is sweeter then it means you are drinking the worst wine. Because worse wines are slightly sweeter than better wines.Here i am not generalising my recommendations based on the taste as there are other characteristics that will determine worse and better wine like full bodied, fruity aroma, astringent, balanced acidity, alcohol, aroma, colour. Besides this, some wines get better when aged as it gets more smother on palate. therefore, in this regard expect i am going to collect more data and proof this.

![](https://cdn-images-1.medium.com/max/800/1*X95-OQhDpKrfa-vuJrTdTw.png)

wordcloud of winetypes and taste

Once my goal was to check how accurate wine taters correctly or accurately predict they type of wine and interestingly 64 % of the time wine reviewers do represent accurately wine-types. Lets see what the actual and predicted value looks like in the following table.

![](https://cdn-images-1.medium.com/max/800/1*cs48m8G-knRuVctcDVLONA.png)

actual vs predicted

> Thank you for reading.
> 
> [You can subscribe to get my essays](https://kirosdsi.com/)that help you to start and grow your startup. Besides you can get the repo[ here.](https://github.com/KirosG/Wine_review_2018/tree/master)

Feel free to connect with me on[linkedin](https://www.linkedin.com/in/kirosg/)or learn more about me[here.](https://kirosg.github.io/)
