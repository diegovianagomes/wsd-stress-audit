# Bibliotecas padrão
import collections
import math
import os
import statistics
import sys
import time
import timeit
from pathlib import Path
from tqdm import tqdm

# Bibliotecas de dados e computação científica
import numpy as np
import pandas as pd

# Bibliotecas estatísticas
import statsmodels.api as sm
from scipy import stats
from scipy.stats import (
    anderson, chi2_contingency, f_oneway, kstest, normaltest,
    shapiro, ttest_rel, wilcoxon, mannwhitneyu
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Bibliotecas de machine learning
import joblib
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, calinski_harabasz_score, cohen_kappa_score,
    confusion_matrix, davies_bouldin_score, f1_score, fbeta_score,
    hamming_loss, jaccard_score, log_loss, matthews_corrcoef,
    max_error, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, precision_score,
    r2_score, recall_score, roc_auc_score, silhouette_score,
    zero_one_loss
)

# Bibliotecas de visualização
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import seaborn as sns

# Bibliotecas utilitárias
from pympler import asizeof

# Sklearn & Imblearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
)
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier