from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, length, when

#initialize Spark session
spark = SparkSession.builder\
    .appName("RecipeCorpusProcessor")\
    .config("spark.driver.memory", "4g")\
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()
#read file
df = spark.read.csv(
    "data/raw/RecipeNLG_dataset.csv", 
    header=True, 
    inferSchema=True,
    multiLine=True,
    escape='"'
)
#drop any rows with null values in the title, ingredients, or instructions columns
clean_df = df.dropna(subset=["title", "ingredients", "directions"])
#normalize the title and instructions columns to lowercase for easier filtering
clean_df = clean_df.withColumn("title_lower", lower(col("title")))
clean_df = clean_df.withColumn("directions_lower", lower(col("directions")))

#filter the dataframe to only include recipes that are relevant to specialized baking and cooking techniques
specialized_df = clean_df.filter(
    col("title_lower").rlike("sourdough boule|batard|sicilian pizza|grandma pizza|garlic knot|banana bread coffee cake|fettuccine") |
    col("ingredients").rlike("psyllium husk|starter discard|00 flour")
)

#add a new column to the dataframe that indicates whether the recipe requires altitude adjustment based on the presence of certain keywords in the instructions
specialized_df = specialized_df.withColumn(
    "requires_altitude_adjustment",
    when(col("directions_lower").rlike("high altitude|elevation"), True).otherwise(False)
)
#add a new column to the dataframe that indicates the length of the directions column
specialized_df = specialized_df.withColumn("direction_char_length", length(col("directions")))
specialized_df = specialized_df.dropDuplicates(["title", "ingredients", "directions"])
#write the specialized dataframe to a parquet file, partitioned by the source column
specialized_df.write \
    .partitionBy("source") \
    .mode("overwrite") \
    .parquet("data/processed/specialized_baking_corpus.parquet")
spark.stop()