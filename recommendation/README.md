# Recommendation System Best Practices

This README captures key points and best practices for building recommendation systems, adopted from the [Microsoft Recommender repo](https://github.com/microsoft/recommenders). There are five key tasks when building a recommendation system:

1. [Prepare Data](#prepare-data): Preparing and loading data for each recommender algorithm
2. [Model](#model): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)).
3. [Evaluate](#evaluate): Evaluating algorithms with offline metrics
4. [Model Select and Optimize](#model-select-and-optimize): Tuning and optimizing hyperparameters for recommender models
5. [Operationalize](#operationalize): Operationalizing models in a production environment on Azure

## Workflow

The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](https://recodatasets.blob.core.windows.net/images/reco_workflow.png)

## 1. Prepare Data

Common data preparation tasks in recommendation system includes data splitting and data transformation. There are multiple **data splitting** protocols (see below) and they should be selected for specific recommendation scenarios.

| Data Split Methods | Description                                                                                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Random             | Randomly assigns entries to either the train set or the test set based on the allocation ratio desired                                                          |
| Stratified         | Split the train and test set based on timestamps by user or item. This accounts for temporal variations when evaluating your model                              |
| Chronological      | In some cases, it is preferable to ensure the same set of users or items are in the train and test sets, this method of splitting will ensure that is the case. |

It is usually observed in the real-world datasets that users may have different types of interactions with items. In addition, same types of interactions (e.g., click an item on the website, view a movie, etc.) may also appear more than once in the history. There are 2 types of interactions (or feedbacks) in recommendation system: explicit and implicit.

- **Explicit feedbacks** are interactions between users and items that result in numerical/ordinal **ratings** or binary preferences such as **like/dislike**
- **Implicit feedbacks** are implicit interactions between users and items (e.g. user may puchase something on a website, click an item on a mobile app, or order food from a restaurant) that reflect users' preference towards the items

Many collaborative filtering algorithms are built on a user-item sparse matrix. This requires that the input data for building the recommender should contain unique user-item pairs.

- For explicit feedback datasets, this can simply be done by deduplicating the repeated user-item-rating tuples.
- For implicit feedback datasets, there are 2 methods to perform the deduplication, depending on the requirements of the actual business user cases.
  - **Data aggregration**: data is aggregated by user to generate affinity scores, which are scores that represent user preferences

| Data Aggregation Methods | Description |
|--------------------------|-------------|
| Count | Count times of interactions between user and item |
| Weighted count | Consider the types of different interactions as weights in the count aggregation. For example, the weights of three different types, "click", "add", and "purchase", can be 1, 2, and 3, respectively |
| Time dependent count | Include a time decay factor in the count aggregation |

  - **Negative sampling**: a technique that samples negative feedback. For example, we can regard the items that a user has not interacted as those that the user does not like.

## 2. Model

Before discussing various models for buliding recommendation systems, it is worthwhile to create a **baseline model** - which is a minimum performance we expect to achieve by a model or starting point used for model comparisons. It is impoty to note that **different baseline approaches should be taken for different problems and business goals**. For example, recommending the previously purchased items could be used as a baseline model for food or restaurant recommendation since people tend to eat the same foods repeatedly. For TV show and/or movie recommendation, on the other hand, recommending previously watched items does not make sense. Probably recommending the most popular (most watched or highly rated) items is more likely useful as a baseline.

There are 2 types of recommendation problems which will be discussed: **rating prediction** and **top-k recommendation**. We use the mean for **rating prediction**, i.e. our baseline model will predict a user's rating of a movie by averaging the ratings the user previously submitted for other movies. For **top-k recommendation** problem, one can use top-k most-rated movies as the baseline model.

| Algorithm | Description |
|-----------|-------------|
| [Alternating Least Squares (ALS)](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html) | Matrix factorization for collaborative filtering problem optimized by the ALS algorithm |
| [Singular Value Decomposition (SVD)](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) | Matrix factorization for collaborative filtering problem optimized by stochastic gradient descent (SGD) |

## 3. Evaluate

Two approaches for evaluating model performance are demonstrated along with their respective metrics.

- **Rating Metrics**: These are used to evaluate how accurate a recommender is at predicting ratings that users gave to items

| Metric | Range | Selection criteria | Description | Reference |
|--------|-------|--------------------|-------------|-----------|
|RMSE|$> 0$|The smaller the better.|Measure of average error in predicted ratings.|[link](https://en.wikipedia.org/wiki/Root-mean-square_deviation)|
|R2|$\leq 1$|The closer to $1$ the better.|Essentially how much of the total variation is explained by the model.|[link](https://en.wikipedia.org/wiki/Coefficient_of_determination)|
|MSE|$\geq 0$|The smaller the better.|Similar to RMSE but uses absolute value instead of squaring and taking the root of the average.|[link](https://en.wikipedia.org/wiki/Mean_absolute_error)|
|Explained variance|$\leq 1$|The closer to $1$ the better.|How much of the variance in the data is explained by the model|[link](https://en.wikipedia.org/wiki/Explained_variation)|

- **Ranking Metrics**: These are used to evaluate how relevant recommendations are for users

| Metric | Range | Selection criteria | Description | Reference |
|-----0--|-------|--------------------|-------------|-----------|
|Precision|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Measures the proportion of recommended items that are relevant.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|Recall|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Measures the proportion of relevant items that are recommended.|[link](https://en.wikipedia.org/wiki/Precision_and_recall)|
|Normalized Discounted Cumulative Gain (NDCG)|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Evaluates how well the predicted items for a user are ranked based on relevance.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|Mean Average Precision (MAP)|$\geq 0$ and $\leq 1$|The closer to $1$ the better.|Average precision for each user normalized over all users.|[link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems)|
|AUC|$\geq 0$ and $\leq 1$|The closer to $1$ the better. 0.5 indicates an uninformative classifier|Depend on the number of recommended items (k).|[link](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)|
|Logloss|$0$ to $\infty$|The closer to $0$ the better.|Logloss can be sensitive to imbalanced datasets.|[link](https://en.wikipedia.org/wiki/Cross_entropy#Relation_to_log-likelihood)|

## 4. Model Select and Optimize

| Notebook | Description |
| --- | --- |
| [tuning_spark_als](https://github.com/kawo123/recommenders/blob/master/notebooks/04_model_select_and_optimize/tuning_spark_als.ipynb) | Step by step tutorials on how to fine tune hyperparameters for Spark based recommender model (illustrated by Spark ALS) with [Spark native construct](https://spark.apache.org/docs/2.3.1/ml-tuning.html) and [`hyperopt` package](http://hyperopt.github.io/hyperopt/). |
| [azureml_hyperdrive_surprise_svd](https://github.com/kawo123/recommenders/blob/master/notebooks/04_model_select_and_optimize/azureml_hyperdrive_surprise_svd.ipynb) | Quickstart tutorial on utilizing [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) for hyperparameter tuning of the matrix factorization method SVD from [Surprise library](https://surprise.readthedocs.io/en/stable/). |

## 5. Operationalize

| Notebook | Description |
| --- | --- |
| [als_movie_o16n](https://github.com/kawo123/recommenders/blob/master/notebooks/05_operationalize/als_movie_o16n.ipynb) | End-to-end examples demonstrate how to build, evaluate, and deploy a Spark ALS based movie recommender with Azure services such as [Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction), and [Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/). |

## References

- [Microsoft Recommenders (GitHub): Best Practices on recommendation systems](https://github.com/microsoft/recommenders)


