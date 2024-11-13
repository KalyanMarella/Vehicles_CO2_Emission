## Vehicles CO2 Emissions Prediction
The Vehicles CO2 Emission Prediction project leverages advanced machine learning and MLOps practices to predict vehicle CO2 emissions based on key vehicle attributes such as engine size, fuel type, and more.

This system uses ZenML to build reproducible and automated ML pipelines, facilitating efficient model training, evaluation, and deployment. MLflow helps track experiments, manage model versions, and ensure transparency in model performance.

The Streamlit interface provides a user-friendly platform for real-time predictions, while model and data drift monitoring ensures long-term accuracy and continuous model improvement.

### Why These Technologies?
- ZenML allows for streamlined automation of the entire ML lifecycle, enhancing reproducibility and scalability.
- MLflow enables robust model tracking and management, ensuring the model remains relevant over time.
- Streamlit creates a seamless user experience for vehicle CO2 emission predictions, making it easy for anyone to interact with the model.
- BentoML: A powerful tool for packaging and deploying machine learning models as production-ready APIs. BentoML helps create reusable and scalable model-serving containers, ensuring fast and efficient deployment of the CO2 emission prediction model.

### Folder Structure and Descriptions
  #### [Steps](https://github.com/KalyanMarella/Vehicles_CO2_Emission/tree/main/steps) :
  - This folder contains the individual steps of the machine learning pipeline. Each script is responsible for a specific task, such as data preprocessing, feature extraction, model training, and
    evaluation. By modularizing the workflow, these steps can be reused or modified independently in the pipeline.
  #### [Pipelines](https://github.com/KalyanMarella/Vehicles_CO2_Emission/tree/main/pipelines) :
  - Training Pipeline: This pipeline automates the training process, including data preprocessing, model selection, hyperparameter tuning, and model evaluation. It ensures that training is consistent,                 reproducible, and scalable using ZenML.
  - Inference Pipeline: Focused on handling incoming data for real-time predictions. This pipeline takes vehicle input data, processes it, and produces CO2 emission predictions. It's optimized for quick 
    response times and ease of deployment.
  #### [app.py](https://github.com/KalyanMarella/Vehicles_CO2_Emission/blob/main/app.py) :
  - The Streamlit app resides here, providing a simple user interface for real-time predictions. Users can input vehicle details and receive CO2 emission predictions, with visualizations to enha
    the user experience.
  #### [Notebooks](https://github.com/KalyanMarella/Vehicles_CO2_Emission/tree/main/notebooks) :
  - This folder contains Jupyter notebooks used for experimentation, data analysis, and visualizations. It helps document the process and test ideas before implementing them in the main pipeline.

### Here is the demo of project interface and model predictions
![Video Preview](notebooks/co2_emission)
