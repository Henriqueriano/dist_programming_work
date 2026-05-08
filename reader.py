import sys
import pandas as pd
from pyspark.sql import *
from lib.logger import Log4j
from lib.utils import *

if __name__ == "__main__":
    conf = get_spark_app_config()

    spark = SparkSession \
        .builder \
        .config(conf=conf) \
        .getOrCreate()

    logger = Log4j(spark)

    if len(sys.argv) != 2:
        sys.exit(-1)

    logger.info("aaaa")

    pdf = pd.read_excel(sys.argv[1])
    df = spark.createDataFrame(pdf)
    
    df_partitioned = df.repartition(2)
    
    result = df_partitioned.selectExpr("sum()")
    result.show()

    logger.info("bbbbb")
    spark.stop()
