import json
import os
import pickle
from typing import Any, List, Type, Union

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "CO2EmissionEnv"


class ListMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (list,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> list:
        """Read from artifact store."""
        list_path = os.path.join(self.uri, 'list.json')
        with fileio.open(list_path, 'r') as f:
            data = json.load(f)
        return data

    def save(self, data: list) -> None:
        """Write to artifact store."""
        list_path = os.path.join(self.uri, 'list.json')
        with fileio.open(list_path, 'w') as f:
            json.dump(data, f)


class SKLearnModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (RegressorMixin, )
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[RegressorMixin]) -> RegressorMixin:
        """Read from artifact store."""
        model_path = os.path.join(self.uri, 'model.joblib')
        return joblib.load(model_path)

    def save(self, model: RegressorMixin) -> None:
        """Write to artifact store."""
        model_path = os.path.join(self.uri, 'model.joblib')
        joblib.dump(model, model_path)