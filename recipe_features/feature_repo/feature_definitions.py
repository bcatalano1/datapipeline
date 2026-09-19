from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import String

#Define the Project
project = Project(name="recipe_features", description="Baking recipe feature repository")

# Define the offline data source pointing to your PySpark output
recipe_source = FileSource(
    name="recipe_parquet_source",
    path="../../data/processed/specialized_baking_corpus.parquet",
    timestamp_field="event_timestamp"
)

recipe_entity = Entity(name = "recipe", join_keys=["recipe_id"])

# Define the Feature View
recipe_feature_view = FeatureView(
    name="recipe_features",
    entities=[recipe_entity],
    ttl=None,
    schema=[
        Field(name="title", dtype=String),
        Field(name="ingredients", dtype=String),
        Field(name="directions", dtype=String),
        Field(name="source", dtype=String),
    ],
    source=recipe_source,
)