# Coastal-Risk-Inequality-Atlas-Europe-
Copernicus coastal flood risk data Eurostat socio-economic indicators OpenStreetMap coastal settlements Python (geopandas, rasterio, pysal)
import numpy as np
import pandas as pd
from scipy.special import softmax

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "income": np.random.normal(30000,8000,n),
    "risk_exposure": np.random.uniform(0,1,n),
    "insurance_cost": np.random.uniform(500,4000,n),
    "relocation_subsidy": np.random.uniform(0,15000,n)
})

# utility functions
u_stay = (
    0.00003*data.income
    -3*data.risk_exposure
    -0.0004*data.insurance_cost
)

u_move = (
    0.00002*data.income
    +0.00008*data.relocation_subsidy
)

utilities=np.vstack([u_stay,u_move])

probs=softmax(utilities,axis=0)

data["prob_move"]=probs[1]

print(data.head())

print("Average relocation probability:",
      data.prob_move.mean())
