# Mastering MLflow for Large Language Models on Databricks

---

**Preface**  
- What to Expect from This Book  
- The Evolving AI & LLM Landscape  
- How to Use This Book & the Public GitHub Repository

---

**Chapter 1: Introduction to MLflow for LLMs on Databricks**  
1.1 Overview of MLflow  
&nbsp;&nbsp;&nbsp;&nbsp;- Components and Capabilities  
&nbsp;&nbsp;&nbsp;&nbsp;- Open Source vs. Databricks Implementations  

1.2 Why MLflow for LLMs?  
&nbsp;&nbsp;&nbsp;&nbsp;- Challenges in LLM Integration  
&nbsp;&nbsp;&nbsp;&nbsp;- How MLflow Alleviates Complexity  

1.3 Introducing the Super Use-Case: HR Policy Chatbot  
&nbsp;&nbsp;&nbsp;&nbsp;- Use-Case Overview  
&nbsp;&nbsp;&nbsp;&nbsp;- Key Requirements and Objectives  

---

**Chapter 2: Core Components of MLflow for LLM Applications**  
2.1 MLflow Tracking  
&nbsp;&nbsp;&nbsp;&nbsp;- Logging Parameters, Metrics, and Artifacts  
&nbsp;&nbsp;&nbsp;&nbsp;- Managing Predictions and Trace Data  

2.2 MLflow Projects  
&nbsp;&nbsp;&nbsp;&nbsp;- Structuring and Running ML Projects (with LLM focus)  

2.3 MLflow Models and the Model Registry  
&nbsp;&nbsp;&nbsp;&nbsp;- Packaging LLMs as PyFuncs  
&nbsp;&nbsp;&nbsp;&nbsp;- Deployment Considerations in the HR Policy Chatbot Context  
&nbsp;&nbsp;&nbsp;&nbsp;- Managing Model Lifecycles and Versioning

---

**Chapter 3: Prompt Engineering with MLflow**  
3.1 The Art and Science of Prompt Engineering  
&nbsp;&nbsp;&nbsp;&nbsp;- Importance for LLM Performance  
&nbsp;&nbsp;&nbsp;&nbsp;- Strategies and Techniques

3.2 MLflow Prompt Engineering UI  
&nbsp;&nbsp;&nbsp;&nbsp;- Features and Capabilities  
&nbsp;&nbsp;&nbsp;&nbsp;- Hands-On Example: Iterating HR Policy Prompts  

3.3 Iterative Development & Optimization  
&nbsp;&nbsp;&nbsp;&nbsp;- Best Practices for Continuous Improvement  
&nbsp;&nbsp;&nbsp;&nbsp;- Tracking Iterations with MLflow

---

**Chapter 4: Evaluating LLMs within MLflow**  
4.1 Understanding LLM Evaluation Challenges  
&nbsp;&nbsp;&nbsp;&nbsp;- Limitations of Traditional Metrics  
&nbsp;&nbsp;&nbsp;&nbsp;- Use-Case Specific Considerations for Chatbots  

4.2 Leveraging the `mlflow.evaluate()` API  
&nbsp;&nbsp;&nbsp;&nbsp;- Standard Metrics for LLMs  
&nbsp;&nbsp;&nbsp;&nbsp;- Custom Scoring Plugins

4.3 Comparative Analysis: Foundational Models, Providers, and Prompts  
&nbsp;&nbsp;&nbsp;&nbsp;- Evaluating Retriever Components  
&nbsp;&nbsp;&nbsp;&nbsp;- Evaluation for Retrieval Augmented Generation (RAG)

---

**Chapter 5: Building a Simple Chain for the HR Policy Chatbot**  
5.1 Architecting a Simple Chain for LLM Applications  
&nbsp;&nbsp;&nbsp;&nbsp;- Overview of the Chain-of-Thought Approach  
&nbsp;&nbsp;&nbsp;&nbsp;- Integrating Tracking and Artifacts

5.2 Implementing MLflow Tracking and Tracing  
&nbsp;&nbsp;&nbsp;&nbsp;- Logging Interactions and Configurations  
&nbsp;&nbsp;&nbsp;&nbsp;- Debugging and Performance Insights

5.3 Evaluating the Retriever and RAG Components  
&nbsp;&nbsp;&nbsp;&nbsp;- Practical Considerations in the Chatbot Use-Case

---

**Chapter 6: Advanced ChatModels, Agents, and Custom PyFuncs**  
6.1 Developing Custom ChatModels with MLflow  
&nbsp;&nbsp;&nbsp;&nbsp;- Wrapping Local LLM Providers  
&nbsp;&nbsp;&nbsp;&nbsp;- Advanced Features: Tracing and Configuration Management

6.2 Implementing Tool-Calling Models and Agents  
&nbsp;&nbsp;&nbsp;&nbsp;- Building Agents with Integrated Tools  
&nbsp;&nbsp;&nbsp;&nbsp;- Case Study: Extending the HR Chatbot with External Data Lookups  
&nbsp;&nbsp;&nbsp;&nbsp;- Evaluating Agent Performance with Custom Metrics

6.3 Packaging and Deploying Advanced LLMs as Custom PyFuncs  
&nbsp;&nbsp;&nbsp;&nbsp;- Code Organization and Best Practices  
&nbsp;&nbsp;&nbsp;&nbsp;- Integration into the HR Policy Chatbot  
&nbsp;&nbsp;&nbsp;&nbsp;- Managing Dependencies and Model Upgrades

---

**Chapter 7: Unified LLM Integrations with MLflow AI Gateway**  
7.1 Overview of the MLflow AI Gateway  
&nbsp;&nbsp;&nbsp;&nbsp;- Simplifying Provider Interactions  
&nbsp;&nbsp;&nbsp;&nbsp;- Key Benefits and Use Cases

7.2 MLflow AI Gateway vs. MosaicAI AI Gateway  
&nbsp;&nbsp;&nbsp;&nbsp;- Feature Comparison and Trade-offs  
&nbsp;&nbsp;&nbsp;&nbsp;- When and Why to Choose One Over the Other

7.3 Provider Swapping and Secure Credential Management  
&nbsp;&nbsp;&nbsp;&nbsp;- Achieving a Consistent API Experience  
&nbsp;&nbsp;&nbsp;&nbsp;- Best Practices for Seamless Integration

---

**Chapter 8: Deep Dive into MLflow Tracing for LLMs**  
8.1 Understanding Tracing in MLflow  
&nbsp;&nbsp;&nbsp;&nbsp;- Concepts and Importance for LLMs  
&nbsp;&nbsp;&nbsp;&nbsp;- Overview of Trace Data and Schemas

8.2 Automated Tracing with GenAI Libraries  
&nbsp;&nbsp;&nbsp;&nbsp;- Integrating with LangChain, OpenAI, LlamaIndex, and AutoGen

8.3 Manual and Low-Level Tracing APIs  
&nbsp;&nbsp;&nbsp;&nbsp;- High-Level Fluent APIs: Decorators and Context Managers  
&nbsp;&nbsp;&nbsp;&nbsp;- Low-Level Client APIs for Fine-Grained Control

8.4 Best Practices for Trace Data Management  
&nbsp;&nbsp;&nbsp;&nbsp;- Debugging, Performance Monitoring, and Insights

---

**Chapter 9: Conclusion and Future Directions**  
9.1 Recap of the HR Policy Chatbot Journey  
&nbsp;&nbsp;&nbsp;&nbsp;- Key Learnings and Outcomes

9.2 Emerging Trends in LLMs and MLflow  
&nbsp;&nbsp;&nbsp;&nbsp;- Future Opportunities and Evolving Use Cases

9.3 Contributing and Engaging with the Open-Source Community  
&nbsp;&nbsp;&nbsp;&nbsp;- Overview of the Public GitHub Repository  
&nbsp;&nbsp;&nbsp;&nbsp;- Next Steps and Final Thoughts

---

**Appendices**  
- Appendix A: GitHub Repository Structure and How to Use It  
- Appendix B: Glossary of Key Terms  
- Appendix C: Additional Resources and References