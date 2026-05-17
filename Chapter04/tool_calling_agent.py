
import os
import yaml
import mlflow

from operator import itemgetter

from databricks_langchain import ChatDatabricks
from databricks_langchain import VectorSearchRetrieverTool

from langchain.agents import create_agent

## Enable MLflow Tracing
mlflow.langchain.autolog()

try:
    with open("chapter04_conf.yml", "r") as f:
        model_config = yaml.safe_load(f)

    with open("uc_model_registry.yml", "r") as f:
        uc_model_conf = yaml.safe_load(f)

except:
    with open("../conf/chapter04_conf.yml", "r") as f:
        model_config = yaml.safe_load(f)

    with open("../conf/uc_model_registry.yml", "r") as f:
        uc_model_conf = yaml.safe_load(f)

databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_tool")
llm_config = model_config.get("llm_config")

# Define LLM
model = ChatDatabricks(
    endpoint=databricks_resources.get("model_name"),
    **llm_config.get("llm_parameters")
)

# Define VS Tool
vector_search_tool = VectorSearchRetrieverTool(
    index_name=retriever_config.get("index_name"),
    num_results=retriever_config.get('num_results'),
    tool_name=retriever_config.get("tool_name"),
    tool_description=retriever_config.get("tool_description"),
    columns=retriever_config.get("columns")
)

# Load System Prompt
prompt_name = uc_model_conf.get("tool_calling_agent_prompt").get("full_name")
promp_template = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@champion").template
system_prompt = model_config.get("system_prompt")

# Create Agent
lc_agent = create_agent(
    model=model,
    tools=[vector_search_tool],
    system_prompt=system_prompt,
)

## Set Model for Models from Code Logging to Work
mlflow.models.set_model(model=lc_agent)
