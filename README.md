# Mastering MLflow for LLMs on Databricks

---

# Preface

- **Why This Book Was Written**
- **What to Expect from This Book**
- **The Evolving AI & LLM Landscape**
- **How to Use This Book & the Public GitHub Repository**

---

# Chapter 1: Introduction to MLflow for LLMs on Databricks

## 1.1 Overview of MLflow
- Why Use MLflow 
- Components and Capabilities
- Open Source vs. Databricks Implementations

## 1.2 Why MLflow for LLMs?
- Challenges in LLM Integration
- How MLflow Alleviates Complexity

## 1.3 Introducing the Super Use-Case: HR Policy Chatbot
- Use-Case Overview
- Key Requirements and Objectives
- Breakdown of the Use Case to Different Chapters

## 1.4 Conclusion

---

# Chapter 2: Core Components of MLflow for LLM Applications

## 2.1 MLflow Tracking
- Logging Parameters, Metrics, and Artifacts
- Managing Experiments

## 2.2 MLflow Projects
- Environment Dependencies  
- Structuring and Running ML Projects 

## 2.3 MLflow Models
- Logging Models with Different Flavors
- Natively Supported Model Flavors
- MLflow Model Artifacts 

## 2.4 Model Registry
- Model Registry in UC 
- Model Versioning
- Managing Model Lifecycles 

## 2.5 Conclusion 

---

# Chapter 3: Prompt Engineering with MLflow

## 3.1 The Art and Science of Prompt Engineering
- Importance for LLM Performance
- Strategies and Techniques of Prompting (Most common ones)

## 3.2 MLflow Prompt Engineering UI
- Features and Capabilities
- Hands-On Example: Iterating HR Policy Prompts

## 3.3 Iterative Development & Optimization
- Best Practices for Continuous Improvement
- Tracking Iterations with MLflow

## 3.4 Conclusion & Best Practices 

---

# Chapter 4: Building and Tracking a Simple Chain

## 4.1 Architecting a Simple Chain for LLM Applications
- Overview of the Chaining Approach
- Integrating with Databricks Foundation Model API
- Integrating Vector Indexes

## 4.2 Implementing MLflow Tracking 
- Logging Interactions and Configurations
- Model Signatures, Input and Output Formats
- Logging Model as Code
- Logging Chain as Code 
- Registering Model in Unity Catalog

## 4.3 Conclusion & Best Practices

---

# Chapter 5: Deep Dive into MLflow Tracing for LLMs

## 5.1 Understanding Tracing in MLflow
- Concepts and Importance for LLMs
- Overview of Trace Data and Schemas
- Debugging and Performance Insights

## 5.2 Automated Tracing with GenAI Libraries
- Integrating with LangChain, OpenAI, LlamaIndex, and AutoGen

## 5.3 Manual and Low-Level Tracing APIs
- High-Level Fluent APIs: Decorators and Context Managers
- Low-Level Client APIs for Fine-Grained Control

## 5.4 Best Practices for Trace Data Management
- Debugging, Performance Monitoring, and Insights

## 5.5 Conclusion & Best Practices

---

# Chapter 6: Evaluating LLMs within MLflow

## 6.1 Understanding LLM Evaluation Challenges
- Limitations of Traditional Metrics
- Components: What can be evaluated (chart)

## 6.2 Offline / Online Evaluation 
- Human Evaluation
- Offline Evaluation
- Online Evaluation

## 6.3 Evaluating the Retriever
- Why Evaluate Retrieval?
- Validation Dataset for Retrieval Evaluation
- Retriever Metrics 

## 6.4 Evaluating the LLM chain
- LLM as a Judge 
- LLM Chains 
- Interpreting Evaluation Results
- RAG Metrics (RAG != Chain)
- Standard Metrics for LLMs
- Custom Scoring Metrics

## 6.5 Conclusion & Best Practices

---

# Chapter 7: Advanced ChatModels, Agents, and Custom PyFuncs

## 7.1 Developing Custom ChatModels with MLflow
- Why and when to use ChatModels? 
- Wrapping Local LLM Providers
- Advanced Features: Tracing, Dependencies and Configuration Management

## 7.2 Implementing Tool-Calling Models and Agents
- Building Agents with Integrated Tools
- Case Study: Extending the HR Chatbot with External Data Lookups
- Tools example: SQL Functions, API Calls, Vector Retrieval…  
- Advanced Features: Tracing, Evaluations  

## 7.3 Packaging and Deploying Advanced LLMs as Custom PyFuncs
- Deploying to Model Serving
- Managing Dependencies and Model Versioning

## 7.4 Conclusion & Best Practices

---

# Chapter 8: Operationalizing LLMs with MLflow

## 8.1 MLflow roles in LLMOps 
- Architecture Flow of MLflow in LLM
- MLflow components in LLMOps  

## 8.2 Overview of the MLflow AI Gateway
- MLflow AI Gateway vs. MosaicAI AI Gateway
- When and Why to Choose One Over the Other

## 8.3 MLFlow Model Registry for Model Ops
- Version control
- Stage / Alias

## 8.4 Monitoring with MLFlow Metrics
- Tracing Statistics 
- Model Evaluation Metrics 

## 8.5 Conclusion & Best Practices

---

# Chapter 9: Conclusion and Future Directions

## 9.1 Recap of the HR Policy Chatbot Journey
- Key Learnings and Outcomes

## 9.2 Emerging Trends in LLMs and MLflow
- Future Opportunities and Evolving Use Cases

## 9.3 Contributing and Engaging with the Open-Source Community
- Overview of the Public GitHub Repository
- Next Steps and Final Thoughts

---

# Appendices

- **Appendix A:** GitHub Repository Structure and How to Use It
- **Appendix B:** Glossary of Key Terms
- **Appendix C:** Additional Resources and References
