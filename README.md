# Practical MLflow for Generative AI on Databricks

**Build High-Quality AI Agents from Prompt Design to Production**

Companion code for the O'Reilly book *Practical MLflow for Generative AI on Databricks*
by Nuwan Ganganath, Julie Nguyen, and Chang Shi Lim (O'Reilly Media, 2026, ISBN 9798341652750).

This repository contains the notebooks, agent code, configuration files, and datasets that accompany the book. It is designed to be worked through as a guided project: run the notebooks in sequence, then adapt the patterns to your own use case.

---

## The running example: Unity Airways

Every chapter builds on a single scenario — a customer-service assistant for a fictional airline, **Unity Airways**. The assistant has to handle the mix most real GenAI applications face: unstructured policy documents, structured booking records, multi-intent user requests, and the need to behave safely under ambiguity. The example evolves chapter by chapter, from a baseline prompt to a fully deployed and monitored agent.

---

## Repository structure

| Path | Book chapter | Contents |
|------|--------------|----------|
| [`Chapter01/`](Chapter01) | Ch 1 — Introduction to MLflow for GenAI on Databricks | Overview (conceptual chapter) |
| [`Chapter02/`](Chapter02) | Ch 2 — End-to-End GenAI Application Lifecycle with MLflow | Overview (conceptual chapter) |
| [`Chapter03/`](Chapter03) | Ch 3 — Prompt Engineering with MLflow | Prompt Registry, evaluation, optimization |
| [`Chapter04/`](Chapter04) | Ch 4 — Building and Versioning a Tool-Calling Agent | LangChain agent + `tool_calling_agent.py` |
| [`Chapter05/`](Chapter05) | Ch 5 — MLflow Tracing for GenAI Application Observability | Automated & manual tracing |
| [`Chapter06/`](Chapter06) | Ch 6 — Evaluating GenAI Applications with MLflow | Evaluation datasets, scorers, human feedback |
| [`Chapter07/`](Chapter07) | Ch 7 — Advanced Agents and Tools | Tools, `ResponsesAgent`, MCP, `agent.py` |
| [`Chapter08/`](Chapter08) | Ch 8 — Deploying a GenAI Application with MLflow | Model Serving, AI Gateway, guardrails, `agent.py` |
| [`Chapter09/`](Chapter09) | Ch 9 — Production Monitoring with MLflow | Online scorers, dashboards, alerts |
| [`Chapter10/`](Chapter10) | Ch 10 — Unifying GenAI Systems with MLflow | Agent Server, MCP, OpenTelemetry |
| [`Appendix/data_ingestion/`](Appendix/data_ingestion) | Supports Ch 4 | Load datasets and build the FAQ vector search index |
| [`conf/`](conf) | — | YAML configuration for chapters, datasets, and registries |
| [`dataset/`](dataset) | — | Unity Airways sample data (Parquet) |
| [`requirements.txt`](requirements.txt) | — | Unified Python dependencies for all chapters |

> **Note:** Chapters 1 and 2 are conceptual and have no notebooks — see the book for the full discussion.

---

## Prerequisites

- Access to a **Databricks workspace**. If you don't have one, sign up for the free edition: <https://www.databricks.com/learn/free-edition>
- **Serverless compute** attached to a notebook.
- A **pay-per-token Foundation Model** served on the Databricks Foundation Model APIs (the examples use `databricks-gpt-oss-120b`).
- A **Databricks Vector Search endpoint** (the config defaults to an endpoint named `vs_endpoint`).
- Familiarity with Python and Databricks notebooks. You do not need to be an MLflow expert.

---

## Getting started

1. **Clone the repository** into your Databricks workspace (via Repos) or locally:
   ```bash
   git clone https://github.com/manganganath/mlflow-genai-book.git
   ```
2. **Install dependencies.** Every notebook installs the unified dependency set in its first cell:
   ```python
   %pip install -r ../requirements.txt
   dbutils.library.restartPython()
   ```
   (Chapters run from their own folder; the Appendix notebooks use `../../requirements.txt`.)
3. **Prepare the data.** Run the notebooks in [`Appendix/data_ingestion/`](Appendix/data_ingestion) **first** to load the Unity Airways datasets and build the FAQ vector search index used from Chapter 4 onward.
4. **Work through the chapters in order** (3 → 10). Each notebook opens with an *About this notebook* cell that maps it to the relevant book chapter and sections.

---

## Configuration

Configuration is separated from code so you can change parameters without editing notebooks. The [`conf/`](conf) folder contains:

| File | Purpose |
|------|---------|
| `data.yml` | Catalog, schema, volume, dataset, table, and vector-index names |
| `chapter04_conf.yml` | Tool-calling agent configuration (Ch 4) |
| `chapter05_conf.yml` | Tracing / retriever configuration (Ch 5) |
| `chapter09_conf.yml` | Monitoring experiment path (Ch 9) |
| `synthetic_eval.yml` | Synthetic evaluation dataset generation settings |
| `uc_model_registry.yml` | Unity Catalog model and prompt registry names |

The defaults use catalog `workspace` and schema `unity_airways`. Edit `conf/data.yml` if you want to use a different Unity Catalog location.

---

## Datasets

Sample Unity Airways data lives in [`dataset/`](dataset) as Parquet files:

- `unity_airways_faq.snappy.parquet` — FAQ documents (unstructured policy content)
- `unity_airways_booking_records.snappy.parquet` — structured booking records
- `unity_airways_sample_qa.snappy.parquet` — sample question/answer pairs for evaluation

---

## Using the code examples

This code is here to help you get your job done. In general, you may use it in your own programs and documentation without asking permission. See the book's *Using Code Examples* section for details. An attribution is appreciated but not required; for example:

> *Practical MLflow for Generative AI on Databricks* by Nuwan Ganganath, Julie Nguyen, and Chang Shi Lim (O'Reilly). Copyright 2026 Nuwan Ganganath, Julie Nguyen, and Chang Shi Lim.

If you have a technical question or a problem using the code examples, please email support@oreilly.com.

---

## License

Released under the [MIT License](LICENSE).
