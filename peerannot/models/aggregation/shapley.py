"""
===================================
Two third agreement
===================================
"""

from ..template import CrowdModel
import numpy as np
import warnings
from pathlib import Path
from tqdm.auto import tqdm


class Shapley(CrowdModel):
    def __init__(self, answers, n_classes=2, **kwargs):
        """Shapley-based weight

        :param answers: Dictionary of workers answers with format
        .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int, optional
        """

        super().__init__(answers)
        self.n_classes = n_classes
        self.sparse = False
        self.n_workers = kwargs["n_workers"]
        if kwargs.get("dataset", None):
            self.path_save = Path(kwargs["dataset"]) / "identification" / "shapley"
        else:
            self.path_save = None
        if kwargs.get("path_remove", None):
            to_remove = np.loadtxt(kwargs["path_remove"], dtype=int)
            self.answers_modif = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    self.answers_modif[i] = val
                    i += 1
            self.answers = self.answers_modif
        if kwargs.get("model", None) is None:
            self.model = "xgboost"
        else:
            if kwargs.get("model") not in ["xgboost", "lightgbm"]:
                raise ValueError("Model not supported")
            self.model = kwargs.get("model")

    def generate_array_from_json(self):
        self.json_data = self.answers
        max_task_index = max(self.json_data.keys())
        max_column_index = max(
            [max(task_data.keys()) for task_data in self.json_data.values()]
        )
        array = np.full((max_task_index + 1, max_column_index + 1), -1)
        for task_index, task_data in self.json_data.items():
            for column, value in task_data.items():
                array[int(task_index), int(column)] = value
        return array

    def prepare_data(self, init=False):  # xgboost and lightgbm dataset
        if init:
            self.y_train = self.MV(self.weight)
            self.X_train_np = self.generate_array_from_json()
        if self.model == "xgboost":
            import xgboost as xgb

            dtrain = xgb.DMatrix(self.X_train_np, label=self.y_train)
            self.dtrain = dtrain
        elif self.model == "lightgbm":
            import lightgbm as lgb

            lgb_train = lgb.Dataset(self.X_train_np, label=self.y_train)
            self.dtrain = lgb_train

    def create_parameters(self, params, gpu_device, **kwargs):
        if self.model == "xgboost":
            params["objective"] = "multi:softmax"
            params["max_depth"] = kwargs["depth"]
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = gpu_device
        elif self.model == "lightgbm":
            ...
            # params['objective'] = 'binary'
            # params['device'] = "gpu"
            # params['gpu_device_id'] = gpu_device
            # params['num_leaves'] = 2**kwargs['depth']
        else:
            raise RuntimeError("Unknown boosting library")
        return params

    def fit_model(self):
        if self.model == "xgboost":
            import xgboost as xgb

            lr = 0.1
            max_bin = 128
            gpu_device = "0"  # specify your GPU (used only for training)
            base_params = {
                "learning_rate": lr,
                "max_bin": max_bin,
                "num_class": self.n_classes,
            }
            params = self.create_parameters(base_params, gpu_device, depth=6)
            bst = xgb.train(
                params,
                self.d_train,
                1000,
                evals=[(self.d_train, "train")],
                verbose_eval=100,
                early_stopping_rounds=20,
            )

        elif self.model == "lightgbm":
            ...
        return bst

    def compute_baseline(self, weight=None):
        """Compute label frequency per task"""
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for worker, vote in task.items():
                baseline[task_id, vote] += weight[int(worker)]
        self.baseline = baseline

    def MV(self, weight=None):
        if not self.sparse:
            self.compute_baseline(weight)
            ans = [
                np.random.choice(
                    np.flatnonzero(self.baseline[i] == self.baseline[i].max())
                )
                for i in range(len(self.answers))
            ]
        else:  # sparse problem
            ans = -np.ones(len(self.answers), dtype=int)
            for task_id in tqdm(self.answers.keys()):
                task = self.answers[task_id]
                count = np.zeros(self.n_classes)
                for w, lab in task.items():
                    count[lab] += weight[int(w) - 1]
                ans[int(task_id)] = int(
                    np.random.choice(np.flatnonzero(count == count.max()))
                )
        return ans

    def get_shapley_weights(self, bst):
        import shap

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(self.X_train)
        return np.mean(np.abs(shap_values), axis=0)

    def run(self, maxiter=100, epsilon=1e-6):
        self.weight = np.ones(len(self.n_workers))
        self.prepare_data(init=True)
        for _ in range(maxiter):
            bst = self.fit_model()
            self.weight_new = self.get_shapley_weights(bst)
            self.ans = self.MV(self.weight)
            if np.linal.norm(self.weight_new - self.weight) < epsilon:
                break
            else:
                self.weight = self.weight_new

    def get_probas(self):
        return self.ans

    def get_answers(self):
        """Argmax of soft labels, in this case corresponds to a majority vote
        with two third consensus.
        If the consensus is not reached, a -1 is used as input.
        Additionally, if a `dataset` path is provided,
        tasks index with a -1 label are saved at
        `<dataset>/identification/twothird/too_hard.txt`

        CLI only: the <dataset> key is the shared input between aggregation
        strategies used as follows
           `peerannot aggregate <dataset> --answers answers.json -s <strategy>`

        :return: Hard labels and None when no consensus is reached
        :rtype: numpy.ndarray
        """
        baseline = self.get_probas()
        self.baseline = baseline
        ans = [
            np.random.choice(np.flatnonzero(self.baseline[i] == self.baseline[i].max()))
            for i in range(len(self.answers))
        ]
        self.ans = ans
        if self.path_save:
            # save shapley values
            ...
            if not self.path_save.exists():
                self.path_save.mkdir(parents=True, exist_ok=True)
            np.savetxt(self.path_save / "shapley.txt", tab, fmt="%1i")
        return np.vectorize(self.converter.inv_labels.get)(np.array(ans))
