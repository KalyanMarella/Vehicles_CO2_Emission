from zenml import step
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently import ColumnMapping
from typing_extensions import Annotated
from typing import Tuple
import pandas as pd
import numpy as np
from zenml.types import HTMLString

@step(enable_cache=False)
def model_drift_detector(
    reference_data: pd.Series,  # Baseline actual values
    current_data: np.ndarray    # New predicted values
) -> Tuple[Annotated[str, "model_drift_report_json"], Annotated[HTMLString, "model_drift_report_html"]]:
    """Detects model drift by comparing baseline and new predicted values using TargetDriftPreset."""

    # Ensure both reference and current data have "actual" and "predicted" columns for compatibility with Evidently
    reference_data_df = pd.DataFrame({"actual": reference_data, "predicted": reference_data}).reset_index(drop=True)
    current_data_df = pd.DataFrame({"actual": reference_data, "predicted": current_data}).reset_index(drop=True)

    # Define Column Mapping for Evidently
    column_mapping = ColumnMapping(target="actual", prediction="predicted")

    # Generate the Model Drift Report using Evidently's TargetDriftPreset
    model_drift_report = Report(metrics=[TargetDriftPreset()])
    model_drift_report.run(reference_data=reference_data_df, current_data=current_data_df, column_mapping=column_mapping)

    # Return JSON and HTML report formats for both pipeline tracking and monitoring
    return model_drift_report.json(), HTMLString(model_drift_report.show(mode="inline").data)
from zenml import step
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently import ColumnMapping
from typing_extensions import Annotated
from typing import Tuple
import pandas as pd
import numpy as np
from zenml.types import HTMLString

@step(enable_cache=False)
def model_drift_detector(
    reference_data: pd.Series,  # Baseline actual values
    current_data: np.ndarray    # New predicted values
) -> Tuple[Annotated[str, "model_drift_report_json"], Annotated[HTMLString, "model_drift_report_html"]]:
    """Detects model drift by comparing baseline and new predicted values using TargetDriftPreset."""

    min_length = min(len(reference_data), len(current_data))
    reference_data = reference_data.iloc[:min_length]
    current_data = current_data[:min_length]

    # Ensure both reference and current data have "actual" and "predicted" columns for compatibility with Evidently
    reference_data_df = pd.DataFrame({"actual": reference_data, "predicted": reference_data}).reset_index(drop=True)
    current_data_df = pd.DataFrame({"actual": reference_data, "predicted": current_data}).reset_index(drop=True)

    # Define Column Mapping for Evidently

    column_mapping = ColumnMapping(target="actual", prediction="predicted")

    # Generate the Model Drift Report using Evidently's TargetDriftPreset
    model_drift_report = Report(metrics=[TargetDriftPreset()])
    model_drift_report.run(reference_data=reference_data_df, current_data=current_data_df, column_mapping=column_mapping)

    # Return JSON and HTML report formats for both pipeline tracking and monitoring
    return model_drift_report.json(), HTMLString(model_drift_report.show(mode="inline").data)
