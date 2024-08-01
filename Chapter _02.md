### Chapter 2: Core Components of MLflow

---

### 2.1 MLflow Tracking

MLflow Tracking is a central component designed to log and query experiments. It allows data scientists to record and organize all the relevant parameters, metrics, and artifacts generated during their machine learning experiments. This facilitates reproducibility and collaboration, making it easier to compare different models and approaches.

#### 2.1.1 Overview of MLflow Tracking

MLflow Tracking captures the essential details of each machine learning run, including:

- **Parameters**: Hyperparameters and configuration settings used in the experiment.
- **Metrics**: Performance measures such as accuracy, precision, recall, and custom metrics.
- **Artifacts**: Output files generated during the run, such as models, plots, and logs.
- **Source Code**: Version and commit information of the code used in the experiment.
- **Tags**: User-defined labels to categorize and filter runs.

#### 2.1.2 Setting Up MLflow Tracking on Databricks

To use MLflow Tracking on Databricks, you can start logging runs directly from your Databricks notebooks. Databricks provides seamless integration, allowing you to focus on your experiments without worrying about setup.

1. **Initialize MLflow**: Start by importing MLflow and initializing a new run.
   ```python
   import mlflow

   mlflow.start_run()
   ```

2. **Log Parameters and Metrics**: Log parameters and metrics using `mlflow.log_param()` and `mlflow.log_metric()`.
   ```python
   mlflow.log_param("learning_rate", 0.01)
   mlflow.log_metric("accuracy", 0.95)
   ```

3. **Log Artifacts**: Save output files and log them as artifacts.
   ```python
   with open("output.txt", "w") as f:
       f.write("Experiment results")
   mlflow.log_artifact("output.txt")
   ```

4. **End Run**: End the run to finalize logging.
   ```python
   mlflow.end_run()
   ```

#### 2.1.3 Querying and Comparing Runs

Once experiments are logged, you can query and compare them using the MLflow UI or the MLflow API. This enables you to identify the best-performing models and track the progress of your experiments over time.

### 2.2 MLflow Projects

MLflow Projects provide a standardized format for packaging and sharing data science code. By defining a project structure and dependencies, MLflow Projects ensure that experiments are reproducible and can be easily shared with others.

#### 2.2.1 Overview of MLflow Projects

An MLflow Project is a directory containing:

- **MLproject File**: A YAML file specifying the project name, environment, and entry points.
- **Source Code**: Scripts and code files required to run the project.
- **Dependencies**: Specifications for the libraries and tools needed.

#### 2.2.2 Creating an MLflow Project

1. **MLproject File**: Create an `MLproject` file to define the project structure.
   ```yaml
   name: MyProject

   conda_env: conda.yaml

   entry_points:
     main:
       command: "python train.py"
   ```

2. **Conda Environment**: Define the environment using a `conda.yaml` file.
   ```yaml
   name: my_env
   channels:
     - defaults
   dependencies:
     - python=3.8
     - scikit-learn
   ```

3. **Source Code**: Include the necessary source code files, such as `train.py`.

#### 2.2.3 Running MLflow Projects

To run an MLflow Project, use the `mlflow run` command:
```bash
mlflow run .
```
This command executes the specified entry point in the `MLproject` file within the defined environment.

### 2.3 MLflow Models

MLflow Models provide a unified format for packaging machine learning models, enabling them to be used across various tools and environments. This standardization simplifies the process of deploying and serving models.

#### 2.3.1 Overview of MLflow Models

MLflow Models support multiple flavors, each representing a different model type and serving environment:

- **Python Function (pyfunc)**: A generic model format that can encapsulate any Python model.
- **H2O**: Models created using the H2O.ai platform.
- **Keras**: Models built with the Keras deep learning framework.
- **MLeap**: Models for deployment in JVM-based environments.
- **ONNX**: Models in the Open Neural Network Exchange format.
- **PMML**: Predictive Model Markup Language models.
- **PyTorch**: Models developed using the PyTorch framework.
- **Scikit-Learn**: Models built with the scikit-learn library.
- **Spark**: Models created with Apache Spark MLlib.
- **TensorFlow**: Models developed using TensorFlow.

#### 2.3.2 Logging and Saving Models

To log and save a model, use the `mlflow.log_model()` function:
```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log model
mlflow.sklearn.log_model(model, "random_forest_model")
```

#### 2.3.3 Loading and Serving Models

To load a logged model, use the `mlflow.load_model()` function:
```python
model_uri = "runs:/<run_id>/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)
```

To serve the model as a REST API, use the `mlflow models serve` command:
```bash
mlflow models serve -m runs:/<run_id>/random_forest_model
```

### 2.4 MLflow Model Registry

The MLflow Model Registry is a centralized repository for managing the lifecycle of MLflow models. It supports versioning, stage transitions, and annotations, providing a collaborative space for data scientists and engineers.

#### 2.4.1 Overview of the Model Registry

The Model Registry organizes models into three key stages:

- **Staging**: Models that are under development and testing.
- **Production**: Models that are deployed and serving predictions.
- **Archived**: Deprecated models that are no longer in use.

#### 2.4.2 Registering Models

To register a model, use the MLflow API to create a new version in the Model Registry:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_uri = "runs:/<run_id>/random_forest_model"
model_name = "RandomForestModel"

client.create_registered_model(model_name)
client.create_model_version(model_name, model_uri, run_id)
```

#### 2.4.3 Managing Model Versions

The Model Registry allows you to transition models between stages and add annotations:
```python
# Transition model version to staging
client.transition_model_version_stage(model_name, version, stage="Staging")

# Add description
client.update_model_version(model_name, version, description="Tested on dataset X")
```

### Conclusion

MLflow's core components—Tracking, Projects, Models, and Model Registry—provide a comprehensive framework for managing the machine learning lifecycle. By integrating these components within Databricks, data scientists and engineers can enhance their workflows, ensuring reproducibility, scalability, and collaboration. This structured approach facilitates efficient experimentation, model development, and deployment, ultimately leading to more robust and reliable machine learning applications.

---

### References

1. MLflow Documentation. (n.d.). Retrieved from [MLflow.org](https://mlflow.org/docs/latest/index.html)
2. Databricks Documentation. (n.d.). Retrieved from [Databricks.com](https://docs.databricks.com/)
3. Hugging Face Transformers Documentation. (n.d.). Retrieved from [Hugging Face](https://huggingface.co/transformers/)
4. PyTorch Documentation. (n.d.). Retrieved from [PyTorch](https://pytorch.org/docs/stable/index.html)
5. Scikit-learn Documentation. (n.d.). Retrieved from [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
6. TensorFlow Documentation. (n.d.). Retrieved from [TensorFlow](https://www.tensorflow.org/api_docs)
