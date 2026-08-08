# Chapter 7: Advanced Agents and Tools

This chapter deep dives into agents for the Unity Airways assistant: designing and
integrating different tools, packaging with `ResponsesAgent`, managing conversation
context, and adding advanced capabilities while keeping agents measurable and governed.

## In this chapter
- Developing Agents with MLflow
- Creating Tools (Vector Retriever, Structured Data Lookup, API-Calling)
- Packaging and Deploying with ResponsesAgent
- Integrating Advanced Capabilities (Context Engineering, MCP Servers as Tools, Multiagents)

## In this folder
- `chapter07_walkthrough.ipynb` — builds the advanced agent, adds retriever,
  structured-data, and API-calling tools, and explores context engineering, MCP, and
  multiagent patterns.
- `agent.py` — the `ResponsesAgent` definition logged as a "model from code".

## Before you start
See the [repository README](../README.md) for prerequisites and setup. Run the
[`Appendix/data_ingestion`](../Appendix/data_ingestion) notebooks first to build the
Unity Airways FAQ vector search index used from Chapter 4 onward.
