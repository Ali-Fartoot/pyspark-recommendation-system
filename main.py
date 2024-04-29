from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, rank, countDistinct, count
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, collect_list


spark_session = SparkSession.builder.appName(
    "video_game_review").config("spark,driver.memry","15g").getOrCreate()

df = (spark_session.read.format("json").options(header="true")
    .load('./Video_Games_5.json'))

df_rec = df.select('reviewerID', 'asin', 'overall').withColumnRenamed("reviewerID","userId")\
                                                                 .withColumnRenamed("asin","itemId")\
                                                                 .withColumnRenamed("overall","rating")
df_rec = df_rec.orderBy("userId", "itemId")

popularity_df = df_rec.groupBy('itemId') \
                 .agg(count('*').alias('popularity')) \
                 .orderBy(col('popularity').desc())

top_popular_items = popularity_df.limit(500)
df_rec_filtered = df_rec.join(top_popular_items, on='itemId', how='inner')
user_window = Window.partitionBy("userId").orderBy(col("itemId").desc())
df_rec_filtered = df_rec_filtered.withColumn("num_items", expr("count(*) over (partition by userId)"))
df_rec_filtered = df_rec_filtered.filter(col("num_items")>=5)

df_rec_filtered.show(vertical=True)
num_unique_items = df_rec_filtered.select('itemId').distinct().count()
print(f"Number of unique items: {num_unique_items}")

num_unique_users = df_rec_filtered.select('userId').distinct().count()
print(f"Number of unique users: {num_unique_users}")


items_per_user = df_rec_filtered.groupBy('userId').count().select('count').toPandas()

percent_items_to_mask = 0.3 
df_rec_final = df_rec_filtered.withColumn("num_items_to_mask", (col("num_items") * percent_items_to_mask).cast("int"))
df_rec_final = df_rec_final.withColumn("item_rank", rank().over(user_window))

indexer_user = StringIndexer(inputCol='userId', outputCol='userIndex').setHandleInvalid("keep")
indexer_item = StringIndexer(inputCol='itemId', outputCol='itemIndex').setHandleInvalid("keep")

df_rec_final = indexer_user.fit(df_rec_final).transform(df_rec_final)
df_rec_final = indexer_item.fit(df_rec_final).transform(df_rec_final)

df_rec_final = df_rec_final.withColumn('userIndex', df_rec_final['userIndex'].cast('integer'))\
               .withColumn('itemIndex', df_rec_final['itemIndex'].cast('integer'))

train_df_rec = df_rec_final.filter(col("item_rank") > col("num_items_to_mask"))
test_df_rec = df_rec_final.filter(col("item_rank") <= col("num_items_to_mask"))

# Configure the ALS model
als = ALS(userCol='userIndex', itemCol='itemIndex', ratingCol='rating',
          coldStartStrategy='drop', nonnegative=True)


param_grid = ParamGridBuilder()\
             .addGrid(als.rank, [1, 20, 30])\
             .addGrid(als.maxIter, [20])\
             .addGrid(als.regParam, [.05, .15])\
             .build()
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3)

model = cv.fit(train_df_rec)

best_model = model.bestModel
print('rank: ', best_model.rank)
print('MaxIter: ', best_model._java_obj.parent().getMaxIter())
print('RegParam: ', best_model._java_obj.parent().getRegParam())
model = als.fit(train_df_rec)

predictions = best_model.transform(test_df_rec)
predictions = predictions.withColumn("prediction", expr("CASE WHEN prediction < 1 THEN 1 WHEN prediction > 5 THEN 5 ELSE prediction END"))

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print(f'Root Mean Squared Error (RMSE): {rmse}')


userRecs = best_model.recommendForAllUsers(100)
user_ground_truth = test_df_rec.groupby('userIndex').agg(collect_list('itemIndex').alias('ground_truth_items'))
user_train_items = train_df_rec.groupby('userIndex').agg(collect_list('itemIndex').alias('train_items'))
user_eval = userRecs.join(user_ground_truth, on='userIndex').join(user_train_items, on='userIndex') \
    .select('userIndex', 'recommendations.itemIndex', 'ground_truth_items', 'train_items', 'recommendations.rating')
user_eval = user_eval.toPandas()
user_eval['itemIndex_filtered'] = user_eval.apply(lambda x:[b for (b,z) in zip(x.itemIndex, x.rating) if b not in x.train_items], axis=1)
user_eval['rating_filtered'] = user_eval.apply(lambda x:[z for (b,z) in zip(x.itemIndex, x.rating) if b not in x.train_items], axis=1)



def score(predicted, actual, metric):
        """
        Parameters
        ----------
        predicted : List
            List of predicted apps.
        actual : List
            List of masked apps.
        metric : 'precision' or 'ndcg'
            A valid metric for recommendation.
        Raises
        -----
        Returns
        -------
        m : float
            score.
        """
        valid_metrics = ['precision', 'ndcg']
        if metric not in valid_metrics:
            raise Exception(f"Choose one valid baseline in the list: {valid_metrics}")
        if metric == 'precision':
            m = np.mean([float(len(set(predicted[:k]) 
                                               & set(actual))) / float(k) 
                                     for k in range(1,len(actual)+1)])
        if metric == 'ndcg':
            v = [1 if i in actual else 0 for i in predicted]
            v_2 = [1 for i in actual]
            dcg = sum([(2**i-1)/math.log(k+2,2) for (k,i) in enumerate(v)])
            idcg = sum([(2**i-1)/math.log(k+2,2) for (k,i) in enumerate(v_2)])
            m = dcg/idcg
        return m

user_eval['precision'] = user_eval.apply(lambda x: score(x.itemIndex_filtered, x.ground_truth_items, 'precision'), axis=1)
user_eval['NDCG'] = user_eval.apply(lambda x: score(x.itemIndex_filtered, x.ground_truth_items, 'ndcg'), axis=1)

MAP = user_eval.precision.mean()
avg_NDCG = user_eval.NDCG.mean()