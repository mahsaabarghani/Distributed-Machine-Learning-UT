#3-ب
from pyspark.sql.functions import min, max, avg, variance, round
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler 
import pandas as pd
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    startTime = time.time()
    spark = SparkSession\
        .builder\
        .appName("soal-3-aqdam")\
        .getOrCreate()
    path = "customers.csv"
    data = spark.read.csv(path, header=True, inferSchema=True)
    income = "Annual Income (k$)"
    income_df = data.select("Annual Income (k$)")  
    result = income_df.select(
    min(income).alias("Min_Income"),
    max(income).alias("Max_Income"),
    avg(income).alias("Avr_Income"),
    round(variance(income), 3).alias("Var_Income")
    )
    result.show()
#3-ج



    columns = ["Annual Income (k$)", "Spending Score (1-100)"]
    data = data.select(columns)

    assembler = VectorAssembler(inputCols=columns, outputCol="features")

    final_data  = assembler.transform(data)

    scaler = StandardScaler(inputCol="features",  
                            outputCol="scaledFeatures") 
    scalerModel = scaler.fit(final_data) 
    final_data = scalerModel.transform(final_data) 
    final_data.select('scaledFeatures').show(5)

    silhouette_score=[] 
    evaluator = ClusteringEvaluator(predictionCol='prediction', 
                                    featuresCol='scaledFeatures',  
                                    metricName='silhouette',  
                                    distanceMeasure='squaredEuclidean') 
    kmeans = KMeans(featuresCol='scaledFeatures', k = 3) 
    model = kmeans.fit(final_data) 
    centers = model.clusterCenters()
    predictions = model.transform(final_data)
    print(f"Silhouette Score for k=3 is {evaluator.evaluate(predictions)}")

    
    print("Cluster Centers: ") 
    for center in centers: 
        print(center)
    predictions.select(*columns).show(5)


    #3-د
    predictions_pd = predictions.toPandas()
    plt.figure(figsize=(7, 7))
    c = predictions_pd['prediction']
    for cluster_id in c.unique():
        clusterData = predictions_pd[c == cluster_id]
        plt.scatter(clusterData["Annual Income (k$)"],
                    clusterData["Spending Score (1-100)"],
                    label=f'Cluster {cluster_id}', s=50)

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("K-Means")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("kmeans.jpg", dpi=300)

    import os
    print(os.getcwd())
    spark.stop() 
    endTime = time.time()
    totalTime = endTime - startTime
    print(f"Total time: {totalTime:} seconds")