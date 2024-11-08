from zenml.client import Client
from zenml import step
import pandas as pd
import numpy as np

@step
def load_baseline_x_data() -> pd.DataFrame:
    # Load the specific artifact version by its unique ID
    artifact = Client().get_artifact_version("4efe4eaf-a5a2-43b0-9295-f674d1ab6ba6")
    baseline_data = artifact.load()
    return baseline_data

@step
def load_baseline_y_data()->pd.Series:
    # Load the specific artifact version by its unique ID
    artifact = Client().get_artifact_version("196241bd-b101-4cfe-9517-994887d3ab09")
    baseline_data = artifact.load()
    return baseline_data