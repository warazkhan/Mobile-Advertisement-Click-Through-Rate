# 📊 Mobile Advertisement Click-Through Rate (CTR) Analysis

This project analyzes **Click-Through Rate (CTR)** data to improve ad targeting and optimize online advertising strategies. The dataset includes details about **ad placement, site/app usage, device type, and network connections**, helping us uncover insights into what factors influence user clicks.

## 🚀 Project Overview
- **Dataset**: 1,048,576 records on mobile advertisements and user interactions.
- **Objective**:
  - Perform **Exploratory Data Analysis (EDA)** to uncover CTR trends.
  - Preprocess data (handling missing values, encoding categorical features).
  - Train and evaluate **machine learning models** to predict CTR.
  - Optimize ad targeting using insights from model predictions.

## 🔍 Dataset Overview
The dataset contains multiple categorical and numerical features:

| Column Name     | Description |
|----------------|-------------|
| `id`           | Unique identifier for each record |
| `click`        | Target variable (1 = clicked, 0 = not clicked) |
| `hour`         | Timestamp (hour of ad display) |
| `C1`           | Anonymous categorical feature |
| `banner_pos`   | Position of the ad banner on the screen |
| `site_id`, `site_domain`, `site_category` | Identifiers for the site where the ad was displayed |
| `app_id`, `app_domain`, `app_category` | Identifiers for the mobile app where the ad appeared |
| `device_id`, `device_ip`, `device_model` | Device identifiers (hashed for anonymity) |
| `device_type`  | Type of device (mobile, tablet, etc.) |
| `device_conn_type` | Type of internet connection (Wi-Fi, cellular, etc.) |
| `C14 - C21`    | Anonymous categorical features |

## 📈 Key Analysis & Insights
1️⃣ **Exploratory Data Analysis (EDA)**  
   - CTR trends analyzed based on **hour of ad display, banner position, site/app category, and device type**.
   - Feature correlation analysis to identify impactful variables.

2️⃣ **Machine Learning Models & Performance (RMSE)**
   - **Lower RMSE indicates better model performance.**
   
   | Model                        | RMSE Score  |
   |------------------------------|------------|
   | 🏆 **XGBRegressor**          | **0.3630**  |
   | 🔥 GradientBoostingRegressor | 0.3635     |
   | 🌲 RandomForestRegressor     | 0.3694     |
   | 📉 LogisticRegression        | 0.4202     |
   | 🌳 DecisionTreeRegressor     | 0.4998     |

3️⃣ **Model Evaluation & Selection**  
   - **XGBoost** had the best predictive performance.
   - **Gradient Boosting & Random Forest** also performed well.
   - **Logistic Regression & Decision Tree** had higher errors, indicating they are less suitable for CTR prediction.

## 🛠️ Tools & Libraries Used
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
