import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


class RFBayesian:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def objective(self, trial):
        """return the f1-score"""

        # search space
        n_estimators = trial.suggest_int("n_estimators", low=100, high=200, step=50)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        min_samples_split = trial.suggest_int(
            "min_samples_split", low=2, high=4, step=1
        )
        min_samples_leaf = trial.suggest_int("min_samples_leaf", low=1, high=5, step=1)
        max_depth = trial.suggest_int("max_depth", low=5, high=7, step=1)
        max_features = trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        )

        # random forest classifier object
        rfc = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
        )
        score = cross_val_score(
            estimator=rfc,
            X=self.X_train,
            y=self.y_train,
            scoring="f1_micro",
            cv=5,
            n_jobs=-1,
        ).mean()

        return score

    def run(self):
        # create a study (aim to maximize score)
        study = optuna.create_study(sampler=TPESampler(), direction="maximize")

        # perform hyperparamter tuning (while timing the process)
        time_start = time.time()
        study.optimize(self.objective, n_trials=100)
        time_bayesian = time.time() - time_start

        # store result in a data frame
        values_bayesian = [
            100,
            study.best_trial.number,
            study.best_trial.value,
            time_bayesian,
        ]
        columns = [
            "Number of iterations",
            "Iteration Number of Optimal Hyperparamters",
            "Score",
            "Time Elapsed (s)",
        ]

        return pd.DataFrame([values_bayesian], columns=columns)
