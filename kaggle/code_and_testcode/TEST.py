from pyspark import SparkContext
import pandas as pd
from pyspark.sql import HiveContext

#初始化
sc = SparkContext()
hiveCtx = HiveContext(sc)



