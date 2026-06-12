"""
Modulo 7: Evaluacion y Optimizacion Avanzada
Script 4: Analisis de Desbalance, Optimizacion de Hiperparametros y Calibracion de Modelos
------------------------------------------------------------------------------------------------
Este script implementa un pipeline completo de MLOps con toma de decisiones conectadas:
1. Carga y submuestreo de la base de datos para preservar todos los fraudes de tarjeta de credito.
2. Competencia de Modelos Base (Logistic Regression, Random Forest, Gradient Boosting) con control de desbalanceo.
3. Seleccion automatica del algoritmo ganador de la competencia.
4. Optimizacion con Nested Cross-Validation (Random Search vs Optuna) exclusivamente sobre el algoritmo ganador.
5. Ajuste final de hiperparametros del algoritmo ganador y entrenamiento del Modelo Optimizado Final.
6. Test de robustez por subgrupo (Monto Alto vs Bajo) con Kolmogorov-Smirnov y PSI sobre el modelo final.
7. Analisis de interpretabilidad del modelo final con Feature Importance (guardada en PNG) y SHAP beeswarm plot (guardada en PNG).
8. Calibracion de probabilidades mediante Isotonic Regression evaluada con Brier Score sobre el modelo final.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import optuna
import shap

# Ajuste estetico de graficos
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 14,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})

# Funcion para calcular el Population Stability Index (PSI)
def calculate_psi(expected, actual, num_buckets=10):
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)
    
    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)
    
    exp_pct = np.where(exp_pct == 0, 0.0001, exp_pct)
    act_pct = np.where(act_pct == 0, 0.0001, act_pct)
    
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi_val

# 1. CARGA Y DESCARGA DE DATOS
print("==================================================")
print("1. CARGANDO Y PREPARANDO DATOS (CREDIT CARD FRAUD)")
print("==================================================")

workspace_dir = r"c:\Users\ANDY\OneDrive\Desktop\Clases dictadas\Clases New Horizons\Python para IA y Webscraping\M7 - Evaluacion y optimizacion avanzada"
local_file = os.path.join(workspace_dir, "creditcard.csv")

if not os.path.exists(local_file):
    print("Descargando creditcard.csv desde el repositorio...")
    url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    urllib.request.urlretrieve(url, local_file)
    print("Descarga completada.")

df = pd.read_csv(local_file)
print(f"Dataset original cargado: {len(df)} registros.")
print(f"Distribucion original de clases:\n{df['Class'].value_counts()}")

# 2. SUBMUESTREO ESTRATIFICADO
# Preservamos todos los casos de fraude (492) y tomamos una muestra aleatoria de 10,000 casos no fraude
# para asegurar ejecuciones rapidas y demostraciones eficientes en el aula.
print("\nRealizando submuestreo estratificado para optimizar tiempos de entrenamiento...")
df_fraud = df[df['Class'] == 1]
df_non_fraud = df[df['Class'] == 0].sample(n=10000, random_state=42)
df_sampled = pd.concat([df_fraud, df_non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset de trabajo creado: {len(df_sampled)} registros.")
print(f"Distribucion de clase en el dataset de trabajo:\n{df_sampled['Class'].value_counts()}")

# Definir variables independientes y dependientes
X = df_sampled.drop(['Time', 'Class'], axis=1)
y = df_sampled['Class']

# Division entrenamiento-test estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Dimensiones de entrenamiento: {X_train.shape} | Dimensiones de test: {X_test.shape}")

# 3. COMPETENCIA DE MODELOS BASE (MODERANDO DESBALANCEO)
print("\n==================================================")
print("2. COMPETENCIA DE MODELOS BASE (SELECCION DEL ALGORITMO)")
print("==================================================")

# A. Logistic Regression con pesos de clase
print("Entrenando Logistic Regression...")
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# B. Random Forest con pesos de clase
print("Entrenando Random Forest...")
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# C. Gradient Boosting utilizando pesos de muestra calculados
print("Entrenando Gradient Boosting...")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluacion
base_models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

results = {}
print("\nResultados de modelos base en el dataset de test:")
for name, model in base_models.items():
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, probs)
    ap_score = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    
    results[name] = {
        'ROC AUC': auc_score,
        'PR AUC': ap_score,
        'F1-Score': f1,
        'MCC': mcc,
        'Brier Score': brier,
        'Probs': probs
    }
    
    print(f" - {name}:")
    print(f"   ROC AUC: {auc_score:.4f} | PR AUC: {ap_score:.4f} | F1-Score: {f1:.4f} | MCC: {mcc:.4f} | Brier: {brier:.4f}")

# Determinar el algoritmo ganador basado en el MCC
winning_name = max(results, key=lambda k: results[k]['MCC'])
print(f"\nAlgoritmo Ganador seleccionado para optimizacion hiperparametrica: {winning_name}")

# 4. CONFIGURACION DINAMICA DEL ESPACIO DE BUSQUEDA DEL GANADOR
if winning_name == 'Logistic Regression':
    model_class = LogisticRegression
    base_params = {'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 42}
    param_dist = {
        'C': [0.01, 0.1, 1.0, 10.0]
    }
    use_sample_weight = False
    
    def get_optuna_params(trial):
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True)
        }
elif winning_name == 'Random Forest':
    model_class = RandomForestClassifier
    base_params = {'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7]
    }
    use_sample_weight = False
    
    def get_optuna_params(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 7)
        }
else:  # Gradient Boosting
    model_class = GradientBoostingClassifier
    base_params = {'random_state': 42}
    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7]
    }
    use_sample_weight = True
    
    def get_optuna_params(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 7)
        }

# Funcion helper para entrenar y evaluar combinaciones de parametros respetando los pesos
def train_and_eval(model_class, params, X_tr, y_tr, X_va, y_va):
    model = model_class(**params, **base_params)
    if use_sample_weight:
        sw = compute_sample_weight(class_weight='balanced', y=y_tr)
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, probs)

# 5. OPTIMIZACION CON NESTED CV: RANDOM SEARCH VS OPTUNA (SOBRE EL GANADOR)
print("\n==================================================")
print(f"3. NESTED CROSS-VALIDATION SOBRE EL GANADOR ({winning_name})")
print("==================================================")

# Estructura de Cross-Validation (3 splits externos y 3 internos)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# A. nested CV con Random Search
print("Ejecutando Nested CV con Random Search...")
start_rs = time.time()

# Crear combinaciones discretas para Random Search a partir de param_dist
if winning_name == 'Logistic Regression':
    param_list = [{'C': c} for c in param_dist['C']]
else:
    param_list = []
    for n in param_dist['n_estimators']:
        for d in param_dist['max_depth']:
            param_list.append({'n_estimators': n, 'max_depth': d})
random.seed(42)
random.shuffle(param_list)
param_list = param_list[:4]

rs_outer_scores = []
for train_outer_idx, test_outer_idx in outer_cv.split(X_train, y_train):
    X_train_out = X_train.iloc[train_outer_idx]
    y_train_out = y_train.iloc[train_outer_idx]
    X_test_out = X_train.iloc[test_outer_idx]
    y_test_out = y_train.iloc[test_outer_idx]
    
    best_inner_score = -1
    best_inner_params = None
    for params in param_list:
        inner_scores = []
        for train_inner_idx, test_inner_idx in inner_cv.split(X_train_out, y_train_out):
            X_train_in = X_train_out.iloc[train_inner_idx]
            y_train_in = y_train_out.iloc[train_inner_idx]
            X_test_in = X_train_out.iloc[test_inner_idx]
            y_test_in = y_train_out.iloc[test_inner_idx]
            
            score = train_and_eval(model_class, params, X_train_in, y_train_in, X_test_in, y_test_in)
            inner_scores.append(score)
        mean_inner = np.mean(inner_scores)
        if mean_inner > best_inner_score:
            best_inner_score = mean_inner
            best_inner_params = params
            
    outer_score = train_and_eval(model_class, best_inner_params, X_train_out, y_train_out, X_test_out, y_test_out)
    rs_outer_scores.append(outer_score)

time_rs = time.time() - start_rs
mean_rs_score = np.mean(rs_outer_scores)
print(f"Random Search Nested CV ROC AUC: {mean_rs_score:.4f} | Tiempo: {time_rs:.2f} segundos")

# B. nested CV con Optuna
print("\nEjecutando Nested CV con Optuna...")
start_optuna = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)

optuna_outer_scores = []
for fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X_train, y_train)):
    X_train_out = X_train.iloc[train_outer_idx]
    y_train_out = y_train.iloc[train_outer_idx]
    X_test_out = X_train.iloc[test_outer_idx]
    y_test_out = y_train.iloc[test_outer_idx]
    
    def objective(trial):
        params = get_optuna_params(trial)
        inner_scores = []
        for train_inner_idx, test_inner_idx in inner_cv.split(X_train_out, y_train_out):
            X_train_in = X_train_out.iloc[train_inner_idx]
            y_train_in = y_train_out.iloc[train_inner_idx]
            X_test_in = X_train_out.iloc[test_inner_idx]
            y_test_in = y_train_out.iloc[test_inner_idx]
            
            score = train_and_eval(model_class, params, X_train_in, y_train_in, X_test_in, y_test_in)
            inner_scores.append(score)
        return np.mean(inner_scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=4)
    
    best_params = study.best_params
    outer_score = train_and_eval(model_class, best_params, X_train_out, y_train_out, X_test_out, y_test_out)
    optuna_outer_scores.append(outer_score)

time_optuna = time.time() - start_optuna
mean_optuna_score = np.mean(optuna_outer_scores)
print(f"Optuna Nested CV ROC AUC: {mean_optuna_score:.4f} | Tiempo: {time_optuna:.2f} segundos")

# Comparar los resultados de Nested CV para elegir el optimizador ganador
if mean_optuna_score > mean_rs_score:
    best_optimizer = "Optuna"
else:
    best_optimizer = "Random Search"
print(f"\nOptimizador Ganador de la etapa de Nested CV: {best_optimizer}")

# C. OPTIMIZACION FINAL EN TODO EL SET DE ENTRENAMIENTO USANDO EL OPTIMIZADOR GANADOR
if best_optimizer == "Optuna":
    print(f"\nEjecutando optimizacion final con Optuna sobre el training set para el modelo {winning_name}...")
    def final_objective(trial):
        params = get_optuna_params(trial)
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            score = train_and_eval(model_class, params, X_tr, y_tr, X_va, y_va)
            scores.append(score)
        return np.mean(scores)

    final_study = optuna.create_study(direction='maximize')
    final_study.optimize(final_objective, n_trials=4)
    best_params_winner = final_study.best_params
else:
    print(f"\nEjecutando optimizacion final con Random Search sobre el training set para el modelo {winning_name}...")
    best_final_score = -1
    best_params_winner = None
    for params in param_list:
        scores = []
        for train_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            score = train_and_eval(model_class, params, X_tr, y_tr, X_va, y_va)
            scores.append(score)
        mean_score = np.mean(scores)
        if mean_score > best_final_score:
            best_final_score = mean_score
            best_params_winner = params

print(f"Mejores parametros encontrados para el modelo ganador ({winning_name}) usando {best_optimizer}: {best_params_winner}")

# D. ENTRENAMIENTO DEL MODELO OPTIMIZADO FINAL
print(f"\nEntrenando modelo {winning_name} final con los parametros optimos en todo el training set...")
optimized_winning_model = model_class(**best_params_winner, **base_params)
if use_sample_weight:
    sw = compute_sample_weight(class_weight='balanced', y=y_train)
    optimized_winning_model.fit(X_train, y_train, sample_weight=sw)
else:
    optimized_winning_model.fit(X_train, y_train)

# Predicciones finales en test y train (para calculo de PSI)
optimized_preds = optimized_winning_model.predict(X_test)
optimized_probs = optimized_winning_model.predict_proba(X_test)[:, 1]
train_probs = optimized_winning_model.predict_proba(X_train)[:, 1]

# Evaluar el modelo optimizado final en test
opt_auc = roc_auc_score(y_test, optimized_probs)
opt_ap = average_precision_score(y_test, optimized_probs)
opt_f1 = f1_score(y_test, optimized_preds)
opt_mcc = matthews_corrcoef(y_test, optimized_preds)
opt_brier = brier_score_loss(y_test, optimized_probs)

print(f"\nResultados del modelo {winning_name} OPTIMIZADO en test:")
print(f" - ROC AUC: {opt_auc:.4f} | PR AUC: {opt_ap:.4f} | F1-Score: {opt_f1:.4f} | MCC: {opt_mcc:.4f} | Brier Score: {opt_brier:.4f}")

# Calcular PSI del modelo (Train vs Test) para ver estabilidad del Score
score_psi = calculate_psi(train_probs, optimized_probs)
print(f"PSI de Predicciones/Scores (Train vs Test): {score_psi:.4f}")
if score_psi >= 0.25:
    print(" -> Alerta: Desviacion critica (Drift) en la distribucion de scores entre entrenamiento y prueba.")
elif score_psi >= 0.10:
    print(" -> Alerta: Desviacion moderada en la distribucion de scores.")
else:
    print(" -> Estabilidad: La distribucion de predicciones es altamente estable.")

# 6. TEST DE ROBUSTEZ POR SUBGRUPO (KOLMOGOROV-SMIRNOV Y PSI) SOBRE EL MODELO FINAL
print("\n==================================================")
print("4. EVALUACION DE SUBGRUPOS (SOBRE MODELO GANADOR OPTIMIZADO)")
print("==================================================")

# Dividir el conjunto de test en dos subgrupos en base a la variable Amount
high_amount_threshold = 100.0
subgroup_high_mask = X_test['Amount'] > high_amount_threshold
subgroup_low_mask = X_test['Amount'] <= high_amount_threshold

scores_high = optimized_probs[subgroup_high_mask]
scores_low = optimized_probs[subgroup_low_mask]

print(f"Transacciones de Monto Alto (> {high_amount_threshold}): {len(scores_high)} registros.")
print(f"Transacciones de Monto Bajo/Moderado (<= {high_amount_threshold}): {len(scores_low)} registros.")

# Aplicar el test de dos muestras de Kolmogorov-Smirnov sobre las probabilidades (scores)
ks_stat, p_val = stats.ks_2samp(scores_high, scores_low)
print(f"\nTest Kolmogorov-Smirnov entre scores de ambos subgrupos:")
print(f" - Estadistica KS: {ks_stat:.4f} | p-value: {p_val:.4e}")

if p_val < 0.05:
    print("Resultado: Se rechaza H0. Las distribuciones de los scores asignados difieren significativamente entre ambos subgrupos.")
else:
    print("Resultado: No se puede rechazar H0. Las distribuciones de los scores asignados son similares.")

# Calcular el PSI entre subgrupos
subgroup_psi = calculate_psi(scores_low, scores_high)
print(f"PSI entre subgrupos (Monto Bajo vs Monto Alto): {subgroup_psi:.4f}")
if subgroup_psi >= 0.25:
    print(" -> Alerta: Desviacion distributiva critica entre transacciones de alto y bajo valor.")
elif subgroup_psi >= 0.10:
    print(" -> Alerta: Desviacion moderada entre transacciones de alto y bajo valor.")
else:
    print(" -> Estabilidad: Distribucion estable entre ambos subgrupos.")

# 7. FEATURE IMPORTANCE Y SHAP VALUES SOBRE EL MODELO FINAL (CON GUARDADO DE GRAFICOS)
print("\n==================================================")
print("5. IMPORTANCIA DE VARIABLES Y SHAP VALUES (GRAFICOS PNG)")
print("==================================================")

# A. Importancia global nativa y grafico de barras
print("Calculando Importancia de Variables Nativa...")
if hasattr(optimized_winning_model, 'feature_importances_'):
    importances = optimized_winning_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
elif hasattr(optimized_winning_model, 'coef_'):
    importances = np.abs(optimized_winning_model.coef_[0])
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Top 5 variables mas importantes (Nativa):")
print(feat_imp.head(5))

# Guardar Grafico de Barras de Importancia Nativa
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh', color='teal')
plt.gca().invert_yaxis()
plt.title(f"Top 10 Importancia de Variables - {winning_name} Optimizado")
plt.xlabel("Importancia")
plt.ylabel("Variables")
plt.tight_layout()
feat_imp_path = os.path.join(workspace_dir, "08_feature_importance.png")
plt.savefig(feat_imp_path, dpi=150)
plt.close()
print(f"Grafico de importancia de variables guardado exitosamente como: {feat_imp_path}")

# B. Importancia con SHAP Values y grafico Beeswarm
print("\nCalculando SHAP Values en una muestra de test (100 registros)...")
explainer = shap.Explainer(optimized_winning_model, X_train.iloc[:100])
shap_values = explainer(X_test.iloc[:100])

if len(shap_values.shape) == 3:
    # Para Random Forest (3D), seleccionamos la clase positiva (indice 1)
    shap_val_class = shap_values[:, :, 1]
    shap_vals_matrix = shap_values.values[:, :, 1]
else:
    shap_val_class = shap_values
    if hasattr(shap_values, 'values'):
        shap_vals_matrix = shap_values.values
    else:
        shap_vals_matrix = shap_values

mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
shap_imp = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)

print("Top 5 variables mas importantes basadas en SHAP (Media Absoluta):")
print(shap_imp.head(5))

# Guardar Grafico SHAP Beeswarm
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_val_class, show=False)
plt.title(f"SHAP Beeswarm Plot - {winning_name} Optimizado", fontsize=12, pad=15)
plt.tight_layout()
shap_beeswarm_path = os.path.join(workspace_dir, "09_shap_beeswarm.png")
plt.savefig(shap_beeswarm_path, dpi=150)
plt.close()
print(f"Grafico SHAP Beeswarm guardado exitosamente como: {shap_beeswarm_path}")

# 8. CALIBRACION DE PROBABILIDADES SOBRE EL MODELO FINAL
print("\n==================================================")
print("6. CALIBRACION DE PROBABILIDADES (SOBRE MODELO GANADOR OPTIMIZADO)")
print("==================================================")

print(f"Brier Score del modelo optimizado antes de calibrar: {opt_brier:.5f}")

# Entrenar un calibrador isotonico
print("Ajustando calibrador isotonico sobre el modelo optimizado...")
calibrated_clf = CalibratedClassifierCV(estimator=optimized_winning_model, method='isotonic', cv=3)
calibrated_clf.fit(X_train, y_train)

# Prediccion con probabilidades calibradas
calibrated_probs = calibrated_clf.predict_proba(X_test)[:, 1]
brier_after = brier_score_loss(y_test, calibrated_probs)

print(f"Brier Score del modelo optimizado despues de calibrar: {brier_after:.5f}")
improvement = opt_brier - brier_after
percent_imp = (improvement / opt_brier) * 100
print(f"Mejora absoluta en Brier Score: {improvement:.5f} ({percent_imp:.2f}% de reduccion del error)")
print("==================================================")
