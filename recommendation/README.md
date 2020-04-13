# Recommendation System Best Practices

This README captures key points and best practices for building recommendation systems, adopted from the [Microsoft Recommender repo](https://github.com/microsoft/recommenders). There are five key tasks when building a recommendation system:

1. [Prepare Data](notebooks/01_prepare_data): Preparing and loading data for each recommender algorithm
2. [Model](notebooks/02_model): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)).
3. [Evaluate](notebooks/03_evaluate): Evaluating algorithms with offline metrics
4. [Model Select and Optimize](notebooks/04_model_select_and_optimize): Tuning and optimizing hyperparameters for recommender models
5. [Operationalize](notebooks/05_operationalize): Operationalizing models in a production environment on Azure

## Workflow

The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](https://recodatasets.blob.core.windows.net/images/reco_workflow.png)

## Prepare Data

Common data preparation tasks in recommendation system includes data splitting and data transformation. There are multiple **data splitting** protocols (see below) and they should be selected for specific recommendation scenarios.

| Data Split Methods | Description                                                                                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Random             | Randomly assigns entries to either the train set or the test set based on the allocation ratio desired                                                          |
| Stratified         | Split the train and test set based on timestamps by user or item. This accounts for temporal variations when evaluating your model                              |
| Chronological      | In some cases, it is preferable to ensure the same set of users or items are in the train and test sets, this method of splitting will ensure that is the case. |

It is usually observed in the real-world datasets that users may have different types of interactions with items. In addition, same types of interactions (e.g., click an item on the website, view a movie, etc.) may also appear more than once in the history. There are 2 types of interactions (or feedbacks) in recommendation system: explicit and implicit. 

- Explicit feedbacks are interactions between users and items that result in numerical/ordinal **ratings** or binary preferences such as **like/dislike**
- Implicit feedbacks are implicit interactions between users and items (e.g. user may puchase something on a website, click an item on a mobile app, or order food from a restaurant) that reflect users' preference towards the items

Many collaborative filtering algorithms are built on a user-item sparse matrix. This requires that the input data for building the recommender should contain unique user-item pairs.

- For explicit feedback datasets, this can simply be done by deduplicating the repeated user-item-rating tuples.
- For implicit feedback datasets, there are several methods to perform the deduplication, depending on the requirements of the actual business user cases.
  1. Data aggregration
    - Data is aggregated by user to generate affinity scores, which are scores that represent user preferences

| Data Aggregation Methods | Description                                                                                                                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Count                    | Count times of interactions between user and item                                                                                                                                                     |
| Weighted count           | Consider the types of different interactions as weights in the count aggregation. For example, the weights of three different types, "click", "add", and "purchase", can be 1, 2, and 3, respectively |
| Time dependent count     | Include a time decay factor in the count aggregation                                                                                                                                                  |

  2. Negative sampling
  	- A technique that samples negative feedback. For example, we can regard the items that a user has not interacted as those that the user does not like.

## Model

## Evaluate

## Model Select and Optimize

## Operationalize

## References

- [Microsoft Recommenders (GitHub): Best Practices on recommendation systems](https://github.com/microsoft/recommenders)


