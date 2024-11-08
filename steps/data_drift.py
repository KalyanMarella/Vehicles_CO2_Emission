from zenml import step
from zenml.steps import StepContext
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from typing_extensions import Annotated
from typing import Tuple
import pandas as pd
from zenml.types import HTMLString

@step(enable_cache=False)
def data_drift_detector(
    context: StepContext,
    baseline_data: pd.DataFrame,
    inference_data: pd.DataFrame
) -> Tuple[Annotated[str, "drift_report_json"], Annotated[HTMLString, "drift_report_html"]]:
    """Custom data drift detection step with Evidently.

    Args:
        context: The step context.
        baseline_data: a Pandas DataFrame representing the baseline data.
        inference_data: a Pandas DataFrame representing the new data to compare.

    Returns:
        The Evidently data drift report rendered in JSON and HTML formats.
    """
    reference_data, current_data = baseline_data, inference_data

    # Generate Data Drift Report with Evidently
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)

    # Return the drift report in JSON and HTML formats
    return drift_report.json(), HTMLString(drift_report.show(mode="inline").data)
