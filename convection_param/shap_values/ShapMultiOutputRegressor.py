import numpy as np
import shap
from sklearn.multioutput import MultiOutputRegressor


def build_tree_explainer(model, *args, **kwargs):
    if isinstance(model, MultiOutputRegressor):
        explainer = MultiOutputTreeExplainer(model, *args, **kwargs)
    else:
        explainer = shap.TreeExplainer(model, *args, **kwargs)
    return explainer


class MultiOutputTreeExplainer(shap.Explainer):
    def __init__(self, model, *args, **kwargs):
        assert isinstance(model, MultiOutputRegressor)
        self.explainers = []
        self.expected_value = []
        for estimator in model.estimators_:
            explainer = shap.TreeExplainer(estimator, *args, **kwargs)
            self.explainers.append(explainer)
            self.expected_value.append(explainer.expected_value)

    def shap_values(self, *args, **kwargs):
        shap_values = []
        for explainer in self.explainers:
            shap_values.append(explainer.shap_values(*args, **kwargs))
        return np.array(shap_values)

    def shap_interaction_values(self, *args, **kwargs):
        shap_interaction_values = []
        for explainer in self.explainers:
            shap_interaction_values.append(explainer.shap_interaction_values)
        return np.array(shap_interaction_values)