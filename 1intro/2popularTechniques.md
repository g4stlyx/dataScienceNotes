# Popular Data Science Techniques

some terms

* raw data: cannot be analysed straight away
* raw facts: untouched data you have accumulated and stored
* primary data: data collection

process in sum = raw data => processing => information

## Techniques for Working with Traditional Data

* data pre-processing: e.g. someone write "USA" as their name, removing/marking it is manually is data pre-processing

    * class labeling (categorical vs numerical): e.g. you mark the "cities" in a category so they cannot put "Sefa" there.

    * data cleansing: removing "Nwe York" and putting "New York" instead

    * dealing with missing values: when they skip a field like "age". Should you put there the average of the data or should you skip the entire field? handling that.

* case specific

    * balancing: e.g. there should be same amount of women as there is man, so the case would be gender-independent.

    * shuffling: randomizing the data to prevent unwanted patterns and misleading results.

## Techniques for Working with Big Data

* data pre-processing:
    * class labeling: number, text, images, video data, audio data etc.
    * data cleansing:
    * dealing with missing values

* case specific
    * text data mining: the ps of deriving valuable, unstructured data from a text
    * data masking: analysing the info. without compromising private details, securing the confidential info

## Bussiness Intelligence (BI) Techniques

    BI analysis: data skills + buss. knowledge & intuition, explains PAST PERFORMANCE like:
    * what happened?
    * when did it happen?
    * how many units did we sell?

* extracting info and presenting it in the form of
    * metrics = numbers with buss. meaning
        * e.g. the trafic of a page from your website
    * KPIs (KEY performance indicators) = metrics + buss. objectives
        * e.g. the traffic generated only from users who clicked on a link provided in your ad campaign 
    * reports
    * dashboards

#### Real Life Examples of BI
* price optimisation, inventory management etc.


## Techniques for Working with Traditional Methods

* regression: a model used for quantifying causal relationships among the different variables included in your analysis
    * linear regression: drawing a line looking at the dots in a graphic
    * logistic regession: when the only values are 1 or 0
* cluster analysis: grouping dots close to each other, with similar features
* factor analysis: grouping explanatary variables together
* time series: graphics with time in the horizantal line

#### Real Life Examples of Traditional Methods
* UX, sales forecasting etc.

## Machine Learning Techniques

* 4 essantial components: data, model, objective function, optimization algorithm
* while training, dont set rules, just the main goal
    * dont explain how to use a bow, instead say hit the target
    * it will of course take lots of tries
* 3 major types of ML:
    * Supervised Learning: uses labelled data. e.g in the shooting robot example
        * you give the arrow types, target info etc. to the robot
    * Unsupervised Learning: you dont give it anything, even no target.
        * it tries a lot, categorizes arrows
        * better for situations like when you have 1m arrows, you'd'nt sign targets to all of them
        * combining these is a valid option too. you first use the unsupervised, robot labels the arrow types, then you use the supervised
    * reinforcement learning: awarding the robot when it fires better than before

### Most Notable Aprroaches for Each Type of ML
* Supervised Learning:
    * SVMs(Support Vector Machines), NNs(Neutral Networks), DEEP LEARNING, Random Forests, Bayesian Networks
* Unsupervised Learning:
    * k-means
    * DEEP LEARNING: broad practical scope of application(exteremly high accuracy)
* Reinforcement Learning:
    * similar to supervised learning, but instead of minimizing the lost, one maximizes reward

#### Real Life Examples of ML
* fraud detection especially in financial sector,banks; client retention, like rewarding old customers with discounts