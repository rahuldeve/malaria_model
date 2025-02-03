from feature_generation import MorganFP
from feature_selection import CorrelationThreshold
from lightgbm import LGBMClassifier
from ray import tune
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FunctionTransformer, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler


def get_pipeline():
    feature_filters = make_pipeline(
        FunctionTransformer(
            lambda df: df.drop(["inchi", "smiles", "mol"], axis=1, errors="ignore")
        ),
        CorrelationThreshold(0.95),
        VarianceThreshold(0.1),
        StandardScaler(),
    )

    pipeline = make_pipeline(
        make_union(feature_filters, MorganFP(rdkit_mol_col_name="mol")),
        # CrossValidatedFeatureSelector(LGBMClassifier(random_state=42), n_splits=2),
        LGBMClassifier(random_state=42),
    )

    return pipeline


def get_pipeline_param_space():
    pipeline_param_space = {
        "featureunion__pipeline__variancethreshold__threshold": tune.uniform(0.05, 0.3),
        "featureunion__pipeline__correlationthreshold__threshold": tune.uniform(
            0.8, 1.0
        ),
        "featureunion__morganfp__n_bits": tune.choice([512, 1024, 2048, 4096]),
        "lgbmclassifier__objective": "binary",
        "lgbmclassifier__metric": "average_precision",
        "lgbmclassifier__verbosity": -1,
        "lgbmclassifier__boosting_type": "dart",
        "lgbmclassifier__reg_alpha": tune.loguniform(1e-8, 1e-1),
        "lgbmclassifier__reg_lambda": tune.loguniform(1e-8, 1e-1),
        "lgbmclassifier__num_leaves": tune.randint(2, 256),
        "lgbmclassifier__subsample": tune.uniform(0.1, 1),
        "lgbmclassifier__colsample_bytree": tune.uniform(0.1, 1),
        "lgbmclassifier__min_child_samples": tune.randint(5, 100),
        "lgbmclassifier__n_jobs": 4,
        "lgbmclassifier__random_state": 42,
        "lgbmclassifier__scale_pos_weight": tune.qrandint(30, 100, 2),
        "lgbmclassifier__n_estimators": tune.randint(50, 1000),
        "lgbmclassifier__max_depth": tune.randint(5, 50),
        # "lgbmclassifier__early_stopping_rounds": ConstantParam(100)
    }

    return pipeline_param_space
