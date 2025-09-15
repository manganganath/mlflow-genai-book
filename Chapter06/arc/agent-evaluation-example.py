# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Evaluation example

# COMMAND ----------

# MAGIC %md
# MAGIC The following code shows how to call and test Agent Evaluation on previously generated outputs. It returns a dataframe with evaluation scores calculated by LLM judges that are part of Agent Evaluation.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd

examples =  {
    "request": [
        "What is Spark?",
        "How do I convert a Spark DataFrame to Pandas?",
    ],
    "response": [
        "Spark is a data analytics framework.",
        "This is not possible as Spark is not a panda.",
    ],
    "retrieved_context": [ # Optional, needed for judging groundedness.
        [{"doc_uri": "doc1.txt", "content": "In 2013, Spark, a data analytics framework, was open sourced by UC Berkeley's AMPLab."}],
        [{"doc_uri": "doc2.txt", "content": "To convert a Spark DataFrame to Pandas, you can use toPandas()"}],
    ],
    "expected_response": [ # Optional, needed for judging correctness.
        "Spark is a data analytics framework.",
        "To convert a Spark DataFrame to Pandas, you can use the toPandas() method.",
    ]
}

result = mlflow.evaluate(
    data=pd.DataFrame(examples),    # Your evaluation set
    # model=logged_model.model_uri, # If you have an MLFlow model. `retrieved_context` and `response` will be obtained from calling the model.
    model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
)

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(result.tables['eval_results'])
