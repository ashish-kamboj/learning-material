### Q. Difference between RDD, Dataframe and Dataset 
 - [RDDs vs. Dataframes vs. Datasets – What is the Difference?](https://www.analyticsvidhya.com/blog/2020/11/what-is-the-difference-between-rdds-dataframes-and-datasets/)
 - [A Tale of Three Apache Spark APIs: RDDs vs DataFrames and Datasets](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)

---

### Q. What are workers, executors, cores in Spark Standalone cluster? [(here)](https://stackoverflow.com/questions/32621990/what-are-workers-executors-cores-in-spark-standalone-cluster#:~:text=own%20Java%20processes.-,DRIVER,in%20a%20given%20Spark%20job.)

---
### Q. Which compression parquet file uses?
parquet is having default snappy compression

---

### Q. RDD in spark
 - [What is RDD in spark](https://stackoverflow.com/questions/34433027/what-is-rdd-in-spark)
 - [Resilient Distributed Dataset (RDD)](https://databricks.com/glossary/what-is-rdd#:~:text=RDD%20was%20the%20primary%20user,that%20offers%20transformations%20and%20actions.)
 - [RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)

---

### Q. Pandas Vs Spark [(here)](https://towardsdatascience.com/stop-using-pandas-and-start-using-spark-with-scala-f7364077c2e0)
 - By default, Spark is multi-threaded whereas Pandas is single-threaded
 - Spark code can be executed in a distributed way, on a Spark Cluster, whereas Pandas runs on a single machine
 - Spark DataFrame assures you fault tolerance (It's resilient) & pandas DataFrame does not assure it. -> Hence if your data processing got interrupted/failed in between processing then spark can regenerate the failed result set from lineage (from DAG) . Fault tolerance is not supported in Pandas. You need to implement your own framework to assure it.
 - Spark is lazy, which means it will only execute when you collect (ie. when you actually need to return something), and in the meantime it builds up an execution plan and finds the optimal way to execute your code
 - This differs to Pandas, which is eager, and executes each step as it reaches it
 - Spark is also less likely to run out of memory as it will start using disk when it reaches its memory limit

---
