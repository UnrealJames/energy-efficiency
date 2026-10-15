import pandas as pd
from ucimlrepo import fetch_ucirepo

SEED = 42
TARGETS = ["heating_load", "cooling_load"]
NUM = ["compactness", "wall_area", "roof_area", "height", "glazing_area"]
CAT = ["orientation", "glazing_dist"]
THRESH = 30

NAMES = {"X1": "compactness", "X2": "surface_area", "X3": "wall_area", "X4": "roof_area",
         "X5": "height", "X6": "orientation", "X7": "glazing_area", "X8": "glazing_dist",
         "Y1": "heating_load", "Y2": "cooling_load"}

def load_data():
    """Download the UCI data and return it with readable column names."""
    ee = fetch_ucirepo(id=242)
    raw_data = pd.concat([ee.data.features, ee.data.targets], axis=1)
    return raw_data.rename(columns=NAMES)
