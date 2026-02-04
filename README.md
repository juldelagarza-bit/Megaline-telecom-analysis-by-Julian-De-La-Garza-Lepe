# Megaline Project: Telecommunications Plans Comparative Analysis analysis-by-Julian-De-La-Garza-Lepe
S5 tripleten project 
Megaline Project: Telecommunications Plans Comparative Analysis
## 📋 Project Description
This project analyzes customer behavior and profitability of two prepaid plans offered by telecommunications operator Megaline: Surf and Ultimate. The main objective is to determine which plan generates more revenue to optimize the advertising budget allocation.

## 🎯 Objectives
Analyze the behavior of 500 Megaline customers during 2018
Compare service usage (calls, SMS, data) between both plans
Calculate monthly revenue per user and plan
Perform statistical tests to validate significant differences
Provide data-driven recommendations for commercial strategy
📊 Dataset Description
The project works with 5 datasets:

users.csv
User information (500 records)
calls.csv
Call data (137,735 records)
messages.csv
SMS data (76,051 records)
internet.csv
Web session data (104,825 records)
plans.csv
Plan information (2 records)
📋 Plan Specifications
Surf Plan:

Monthly fee: $20
Includes: 500 minutes, 50 SMS, 15 GB
Overages: $0.03/min, $0.03/SMS, $10/GB
Ultimate Plan:

Monthly fee: $70
Includes: 3000 minutes, 1000 SMS, 30 GB
Overages: $0.01/min, $0.01/SMS, $7/GB
🛠️ Technologies and Libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import math as mt
📈 Methodology
Data Preparation
Data cleaning and type conversion
Handling missing values
Creating derived variables
Exploratory Data Analysis
Descriptive statistics by plan
Comparative visualizations
Distribution analysis
Revenue Calculation
Monthly aggregation per user
Overage and additional cost calculation
Total revenue per plan
Statistical Testing
Revenue comparison between plans
Regional analysis (NY-NJ vs other regions)
Significance level: α = 0.05
📊 Key Findings
User Behavior:

Minutes: Similar usage between plans (Surf: 412.1 min, Ultimate: 410.2 min)
Messages: Minimal difference (Surf: 40.1, Ultimate: 46.3)
Data: Comparable consumption (Surf: 16.7 GB, Ultimate: 17.2 GB)
Revenue Analysis:

Surf Plan: Average revenue $58.15/month (190% above base price)
Ultimate Plan: Average revenue $72.17/month (3% above base price)
Market Share: Surf 68.6% vs Ultimate 31.4%
🎯 Business Recommendations
Primary Recommendation: Focus on Surf Plan

Reasons:

Higher Profitability: Generates $38.15 extra revenue per user vs $2.17 for Ultimate
Larger Market Share: 68.6% of customers prefer this plan
Revenue Generation Model: Users consistently exceed limits, creating additional revenue streams
Strategic Considerations:

Consider creating an intermediate plan between Surf and Ultimate
Surf users are willing to pay for additional services
Ultimate plan has generous limits that are rarely exceeded
📈 Statistical Validation
Hypothesis Testing Results:

Plan Comparison: Statistically significant difference in revenue (p < 0.05)
Regional Analysis: No significant difference between NY-NJ and other regions
Confidence Level: 95%

## 5. Structure
```
megaline-analysis/
│
├── README.md                          # Documentación principal del proyecto
├── README_ES.md                       # Documentación en español
│
├── data/                              # Carpeta de datos
│   ├── raw/                          # Datos originales sin procesar
│   │   ├── megaline_calls.csv
│   │   ├── megaline_internet.csv
│   │   ├── megaline_messages.csv
│   │   ├── megaline_plans.csv
│   │   └── megaline_users.csv
│   └── processed/                    # Datos procesados (si los generas)
│
├── notebooks/                        # Jupyter notebooks
│   └── megaline_analysis.ipynb      # Tu notebook principal
│
├── src/                              # Código fuente (opcional)
│   ├── __init__.py
│   ├── data_processing.py
│   ├── analysis.py
│   └── visualization.py
│
├── results/                          # Resultados del análisis
│   ├── figures/                      # Gráficos generados
│   └── reports/                      # Reportes finales
│
├── requirements.txt                  # Dependencias del proyecto
└── .gitignore 
```
## Autor
Julian De La Garza Lepe
