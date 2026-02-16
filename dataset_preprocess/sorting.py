
from dataclasses import dataclass
import numpy as np
from pandas import DataFrame

@dataclass
class AttributeStats:
    attribut_name: str
    type: str  # 'categorical' or 'numerical'
    id_to_ressource: dict | None  # Only for categorical attributes
    new_column_name: str  # Name of the column in DataFrame

    ressource_to_id: dict | None = None  # Only for categorical attributes
    num_categories: int | None = None  # Only for categorical attributes

    min: float = None
    max: float = None
    mean: float = None
    std: float = None
    quantiles: list = None  # List of quantiles [0.0025, 0.25, 0.5, 0.75, 0.9975]
    median: float = None

    missing_rate: float = None  # Proportion of missing values
    normalized: bool = False  # Whether the attribute has been normalized

def _build_cat_stats(attr, attribute_name: str, new_column_name: str, missing_rate: float):
    values = sorted(attr.dropna().astype(str).unique())
    res_to_id = {v: i+1 for i, v in enumerate(values)} # reserve 0 for missing
    id_to_res = {i: v for v, i in res_to_id.items()}
    K = len(res_to_id) # number of real categories

    return AttributeStats(
        attribut_name=attribute_name,
        type="categorical",
        id_to_ressource=id_to_res,
        new_column_name=new_column_name,
        ressource_to_id=res_to_id,
        num_categories=K,
        missing_rate=missing_rate,
        normalized=True,
    ), res_to_id, K


def get_categorial_attributes(
    df: DataFrame,
    attribute_name: str,
    norm_stats: "AttributeStats" = None,
    new_column_name: str = None,
    **kwargs
    ):
    if new_column_name is None:
        new_column_name = attribute_name

    attr = df[attribute_name]
    missing_rate = float(attr.isna().mean())

    print(f"Processing categorical attribute: {attribute_name}, missing rate: {missing_rate}")

    # train: mapping bauen | val/test: mapping wiederverwenden
    if norm_stats is None:
        stats, res_to_id, K = _build_cat_stats(attr, attribute_name, new_column_name, missing_rate)
        # Speichere res_to_id und K in stats, wenn du AttributeStats erweiterst:
        stats.ressource_to_id = res_to_id
        stats.num_categories = K
    else:
        stats = norm_stats
        res_to_id = stats.ressource_to_id
        K = stats.num_categories
    
    mapped = attr.astype(object).where(attr.notna(), None)  # Keep NaNs as None
    mapped = mapped.map(lambda v: str(v) if v is not None else None)

    # mapping unknown or NaNs to 0
    ids = mapped.map(lambda v: res_to_id.get(v, 0)).astype(int).to_numpy()

    x = ids.astype(float)

    # skaliere direkt mit Nenner K: id/K -> [0,1]
    # unknown bleibt 0 -> 0.0
    if K > 0:
        x = x / float(K)
    else:
        x[:] = 0.0

    df[new_column_name] = x

    s = df[new_column_name]
    stats.min, stats.max = float(s.min()), float(s.max())
    stats.mean, stats.std = float(s.mean()), float(s.std())
    stats.median = float(s.median())
    stats.quantiles = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).tolist()

    stats.ressource_to_id["unknown"] = 0  # add unknown mapping
    stats.id_to_ressource[0] = "unknown"

    return df, stats

def get_numerical_attributes(df: DataFrame, attribute_name:str, norm_stats:AttributeStats =None, new_column_name:str=None, **kwargs):
    
    if "normalize" not in kwargs:
        normalize = True
    else:
        normalize = kwargs["normalize"]
    if "norm_function" in kwargs:
        norm_fct = kwargs["norm_function"]
        print("Custom normalization function provided")
    else:
        norm_fct = min_max_normalization
        print("No custom normalization function provided, using min_max_normalization")

    attr = df[attribute_name]
    missing_rate = attr.isna().sum() / len(attr)

    print(f"Processing numerical attribute: {attribute_name}, missing rate: {missing_rate}")
    
    min_attr = attr.min()
    max_attr = attr.max()
    mean_attr = attr.mean()
    std_attr = attr.std()
    quantiles = attr.quantile([0.0025, 0.25, 0.5, 0.75, 0.9975]).tolist()
    median_attr = attr.median()

    attr = attr.where(attr.notna(), 0) #set after calculating metrics, important!

    attribute = AttributeStats(
        attribut_name=attribute_name,
        type="numerical",
        new_column_name=new_column_name,
        min=min_attr,
        max=max_attr,
        mean=mean_attr,
        std=std_attr,
        quantiles=quantiles,
        median=median_attr,
        missing_rate=missing_rate,
        normalized=normalize,
        id_to_ressource=None
    )

    if normalize:
        if norm_stats is not None:
            print("Using provided normalization stats.")
            assert isinstance(norm_stats, AttributeStats), "norm_stats must be an instance of AttributeStats."
            assert norm_stats.min is not None and norm_stats.max is not None, "norm_stats must have min and max values defined."
            min_attr = norm_stats.min
            max_attr = norm_stats.max
            mean_attr = norm_stats.mean
            std_attr = norm_stats.std
            quantiles = norm_stats.quantiles
            median_attr = norm_stats.median
        else:
            print("Using calculated normalization stats. If this is not your training set, this leads to data leakage.")
        df[attribute.new_column_name] = attr.apply(lambda x: quantile_min_max_normalization(x, 
                                                                          quantiles=quantiles))

    return df, attribute

def is_numerical_series(series: DataFrame) -> bool:
    for cell in series.dropna():
        try:
            float(cell)
        except ValueError:
            return False
    return True

def min_max_normalization(x, **kwargs):
    assert "min_val" in kwargs and "max_val" in kwargs, "min_max_normalization requires 'min_val' and 'max_val' as keyword arguments."
    
    min_val = float(kwargs["min_val"])
    max_val = float(kwargs["max_val"])
    den = max_val - min_val

    return 0.0 if den == 0 else (x - min_val) / den

def quantile_min_max_normalization(x, **kwargs):
    """
    Numerisch -> min-max Normalisierung basierend auf Quantilen (1% - 99%)
    kwargs: quantiles (Liste mit Quantilen [1%, 25%, 50%, 75%, 99%])
    """
    assert "quantiles" in kwargs, "quantile_min_max_normalization requires 'quantiles' as keyword argument."
    
    quantiles = kwargs["quantiles"]
    q_01 = float(quantiles[0])
    q_99 = float(quantiles[4])
    den = q_99 - q_01

    return 0.0 if den == 0 else (x - q_01) / den

def tanh_zscore_normalization(x, **kwargs):
    """
    Numerisch -> z-Score (mit Train-Stats) -> tanh-squash nach (-1, 1)
    kwargs: mean, std, (optional) c
    """
    mean = float(kwargs["mean"])
    std = float(kwargs["std"])
    c = float(kwargs.get("c", 4.0))

    if not np.isfinite(std) or std == 0.0:
        return 0.0

    z = (float(x) - mean) / std
    return float(np.tanh(z / c))