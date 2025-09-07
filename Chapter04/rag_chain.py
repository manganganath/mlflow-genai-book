
import os
import mlflow

from operator import itemgetter

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import ConfigurableField

from helpers.format_helpers import (
    extract_user_query_string,
    extract_previous_messages,
    combine_all_messages_for_vector_search,
    format_context,
    format_docs
)

## Enable MLflow Tracing
mlflow.langchain.autolog()

## Get the conf from the local conf file
model_config = mlflow.models.ModelConfig(development_config="../conf/chapter04_conf.yml")
databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

## Vector Search
vector_search_schema = retriever_config.get("retriever_schema")

vector_search_as_retriever = DatabricksVectorSearch(
    index_name=retriever_config.get("index_name"),
    columns=[
        vector_search_schema.get("primary_key"),
        vector_search_schema.get("chunk_text"),
        vector_search_schema.get("document_uri"),
    ],
).as_retriever(search_kwargs=retriever_config.get("parameters"))


mlflow.models.set_retriever_schema(
    primary_key=vector_search_schema.get("primary_key"),
    text_column=vector_search_schema.get("chunk_text"),
    doc_uri=vector_search_schema.get("document_uri"),
)

## Prompt Template
prompt = PromptTemplate(
    template=llm_config.get("llm_prompt_template"),
    input_variables=llm_config.get("llm_prompt_template_variables"),
)

## LLM
model = ChatDatabricks(
    endpoint=databricks_resources.get("model_name"),
    **llm_config.get("llm_parameters")
)

## RAG Chain
chain = (
    {
        "context": itemgetter("messages") | RunnableLambda(extract_user_query_string) | vector_search_as_retriever | RunnableLambda(format_docs),
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string)
    }
    | prompt
    | model
    | StrOutputParser()
)

## Set Model for Models from Code Logging to Work
mlflow.models.set_model(model=chain)
