from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize a local Spark session
spark = SparkSession.builder \
    .appName("BakingDataAnalysis") \
    .getOrCreate()
# Load the optimized Parquet dataset (update the path if your output folder differs)
df = spark.read.parquet("data/processed/specialized_baking_corpus.parquet")
df.createOrReplaceTempView("recipes")

# Display the schema to understand the available structure
print("Dataset Schema:")
df.printSchema()

# # Query example: Filter for sourdough recipes and show the top 5 results
# query = """
#     SELECT 
#         title, 
#         ingredients,
#         directions
#     FROM recipes
#     WHERE (LOWER(title) LIKE '%sourdough%' OR LOWER(title) LIKE '%pizza dough%')
#       AND LOWER(ingredients) LIKE '%flour%'
#       AND LOWER(ingredients) LIKE '%water%'
#     ORDER BY LENGTH(directions) DESC
#     LIMIT 5
# """

# sql_results = spark.sql(query)
# sql_results.show(truncate=False)

# Write the SQL query results back out to a new localized CSV or Parquet
#spark.sql(query).coalesce(1).write.mode("overwrite").csv("data/processed/artisan_recipes_filtered")


window_query = """
    WITH CategorizedRecipes AS (
        SELECT 
            title, 
            ingredients,
            directions,
            CASE 
                WHEN LOWER(title) LIKE '%sourdough%' THEN 'Sourdough'
                WHEN LOWER(title) LIKE '%pizza dough%' THEN 'Pizza Dough'
                ELSE 'Other'
            END AS dough_type,
            LENGTH(directions) AS instruction_length
        FROM recipes
        WHERE LOWER(title) LIKE '%sourdough%' OR LOWER(title) LIKE '%pizza dough%'
    )
    SELECT 
        dough_type,
        title,
        instruction_length,
        RANK() OVER (PARTITION BY dough_type ORDER BY instruction_length DESC) as complexity_rank
    FROM CategorizedRecipes
"""

# Execute the query and filter for the top 3 most complex recipes per category
ranked_df = spark.sql(window_query)
ranked_df.filter(col("complexity_rank") <= 3).show(truncate=False)


# print("Sourdough Extraction:")
# sourdough_df = df.filter(col("title").ilike("%sourdough%"))
# sourdough_df.select("title", "ingredients").show(5, truncate=False)