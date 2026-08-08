# Chapter 8: Deploying a GenAI Application with MLflow

This chapter covers deployment strategies that fit enterprise realities: serving the
Unity Airways agent on Databricks Model Serving, using the MLflow AI Gateway and AI
guardrails, and applying LLM and agent operations for repeatable releases.

## In this chapter
- Deploying to Databricks Model Serving
- MLflow AI Gateway
- AI Guardrails
- LLM Operations and Agent Operations
- Challenges of Productionization

## In this folder
- `chapter08_walkthrough.ipynb` — deploys the agent to Model Serving and demonstrates
  the AI Gateway, guardrails, and operational patterns.
- `agent.py` — the deployable agent definition logged as a "model from code".

## Before you start
See the [repository README](../README.md) for prerequisites and setup. Run the
[`Appendix/data_ingestion`](../Appendix/data_ingestion) notebooks first to build the
Unity Airways FAQ vector search index used from Chapter 4 onward.
