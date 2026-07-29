# Chapter 4: Building and Versioning a Tool-Calling Agent

This chapter builds a tool-calling agent for the Unity Airways assistant and versions
it as a reproducible application artifact, bringing configuration, retrieval, and
prompting together as a governed unit you can compare and promote.

## In this chapter
- What Is a Tool-Calling Agent?
- LangChain and Databricks Integration
- Building a LangChain Agent and Versioning in MLflow
- Packaging and Logging the Agent and Artifacts

## In this folder
- `chapter04_walkthrough.ipynb` — builds the LangChain tool-calling agent
  and logs and versions it with MLflow.
- `tool_calling_agent.py` — the agent definition logged as a "model from code".
- `requirements.txt` — pins the agent's deployment dependencies.

## Before you start
See the [repository README](../README.md) for prerequisites and setup. Run the
[`Appendix/data_ingestion`](../Appendix/data_ingestion) notebooks first to build the
Unity Airways FAQ vector search index used from Chapter 4 onward.
