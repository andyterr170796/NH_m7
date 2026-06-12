"""
Módulo 7: Evaluación y Optimización Avanzada
Script 1: Técnicas para el Manejo de Desbalance de Clases
------------------------------------------------------------------------------------------------
Este script se enfoca exclusivamente en el Tópico 1 de la agenda de clase.
Enseña cómo abordar problemas de la vida real con clases minoritarias extremas y costos de negocio
asimétricos mediante métricas robustas, técnicas de remuestreo (SMOTE/Class Weights), optimización
financiera de umbrales y calibración de probabilidades.

Dataset Utilizado: German Credit Dataset (UCI / OpenML) - Datos de Corte Transversal Estándar.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estética de gráficos de alto nivel
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 14,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})

# 1. CARGA Y PREPARACIÓN DE DATOS (CORTE TRANSVERSAL)
print("==================================================")
print("1. CARGANDO DATASET DE CORTE TRANSVERSAL (GERMAN CREDIT)...")
print("==================================================")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Intentamos cargar el dataset clásico "German Credit" (credit-g) de OpenML
credit_data = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
df = credit_data.frame
print(" -> Dataset German Credit cargado exitosamente desde OpenML.")

# Traducir la variable objetivo a binario: 1 = bad (malo/riesgoso), 0 = good (bueno)
df['target'] = (df['class'] == 'bad').astype(int)
df = df.drop(columns=['class'])

# Forzar un desbalance extremo del 5% para simular casos reales de fraude o morosidad severa
# Distribución original: 70% good, 30% bad
df_good = df[df['target'] == 0]
df_bad = df[df['target'] == 1].sample(n=35, random_state=42) # Reducimos la minoritaria a 35 casos
df_imbalanced = pd.concat([df_good, df_bad]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Distribución de la variable objetivo:")
print(f" - Casos Buenos (0): {len(df_imbalanced[df_imbalanced['target'] == 0])}")
print(f" - Casos Malos (1) [Minoritaria]: {len(df_imbalanced[df_imbalanced['target'] == 1])}")
print(f" - Proporción clase positiva: {df_imbalanced['target'].mean() * 100:.2f}%")

# Separar características y variable objetivo
target_col = 'target'
X = df_imbalanced.drop(columns=[target_col])
y = df_imbalanced[target_col]

# Identificar columnas numéricas y categóricas
cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

# Crear preprocesador
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# 2. ENFOQUES PARA MANEJO DE DESBALANCE (CLASS WEIGHTS VS RESAMPLING)
print("\n==================================================")
print("2. ENTRENANDO ENFOQUES DE DESBALANCE...")
print("==================================================")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, fbeta_score, matthews_corrcoef

# Modelo A: Regresión Logística Estándar (Sin ajuste)
model_standard = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
model_standard.fit(X_train, y_train)

# Modelo B: Regresión Logística con Ponderación de Clases (Class Weights) - Diapositiva 21
model_weighted = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
model_weighted.fit(X_train, y_train)

# Modelo C: Remuestreo Sintético (SMOTE) - Diapositiva 20
print("[INFO] Intentando aplicar SMOTE para sobremuestreo sintético...")
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    model_smote = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model_smote.fit(X_train, y_train)
    has_smote = True
    print(" -> SMOTE entrenado con éxito usando imbalanced-learn.")
except ImportError:
    has_smote = False
    print(" -> [AVISO DOCENTE] 'imbalanced-learn' no está instalado.")
    print("    Para enseñar SMOTE en vivo, pida a los alumnos instalarlo con: pip install imbalanced-learn")
    print("    Como alternativa robusta en este script, usaremos Random Oversampling manual sobre el conjunto de entrenamiento.")
    
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    multiplier = len(neg_idx) // (3 * len(pos_idx))
    pos_idx_oversampled = np.repeat(pos_idx, max(1, multiplier))
    oversampled_train_idx = np.concatenate([neg_idx, pos_idx_oversampled])
    
    X_train_oversampled = X_train_pre[oversampled_train_idx]
    y_train_oversampled = y_train.iloc[oversampled_train_idx]
    
    model_smote = LogisticRegression(max_iter=1000, random_state=42)
    model_smote.fit(X_train_oversampled, y_train_oversampled)


# 3. COMPARATIVA DE MÉTRICAS ROBUSTAS
print("\n==================================================")
print("3. COMPARATIVA DE MÉTRICAS ROBUSTAS (DIAPOSITIVA 18)")
print("==================================================")

def evaluate_predictions(y_true, y_pred, y_prob, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Balanced Accuracy (promedio de recall de ambas clases)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + spec) / 2
    
    # F2-Score (prioriza Recall sobre Precision, crítico en riesgo)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    f1 = fbeta_score(y_true, y_pred, beta=1)
    
    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Curvas AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_vals, prec_vals)
    
    print(f"\n--- Modelo: {name} (Umbral 0.5) ---")
    print(f" Matriz de Confusión: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f" Accuracy:            {acc:.4f} (¡Cuidado! Esta métrica es muy engañosa)")
    print(f" Balanced Accuracy:   {balanced_acc:.4f}")
    print(f" Precision:           {precision:.4f}")
    print(f" Recall / Sensib.:    {recall:.4f}")
    print(f" F1-Score:            {f1:.4f}")
    print(f" F2-Score (Beta=2):   {f2:.4f} (Prioriza Recall/Sensibilidad)")
    print(f" MCC:                 {mcc:.4f}")
    print(f" ROC-AUC:             {roc_auc:.4f} (Inundado por la clase mayoritaria)")
    print(f" PR-AUC:              {pr_auc:.4f} (Sensible a la clase positiva rara)")
    
    return roc_auc, pr_auc

y_prob_std = model_standard.predict_proba(X_test)[:, 1]
y_pred_std = model_standard.predict(X_test)
evaluate_predictions(y_test, y_pred_std, y_prob_std, "Estándar (Sin Ajustes)")

y_prob_w = model_weighted.predict_proba(X_test)[:, 1]
y_pred_w = model_weighted.predict(X_test)
evaluate_predictions(y_test, y_pred_w, y_prob_w, "Con Class Weights")

if has_smote:
    y_prob_res = model_smote.predict_proba(X_test)[:, 1]
    y_pred_res = model_smote.predict(X_test)
else:
    y_prob_res = model_smote.predict_proba(X_test_pre)[:, 1]
    y_pred_res = model_smote.predict(X_test_pre)
evaluate_predictions(y_test, y_pred_res, y_prob_res, "Con Resampling (SMOTE)")

# Gráfico 1: Curva ROC vs Precision-Recall
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Curva ROC
for prob, name in [(y_prob_std, 'Estándar'), (y_prob_w, 'Class Weights'), (y_prob_res, 'SMOTE Resampling')]:
    fpr_vals, tpr_vals, _ = roc_curve(y_test, prob)
    ax1.plot(fpr_vals, tpr_vals, label=f'{name} (AUC = {auc(fpr_vals, tpr_vals):.2f})', lw=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('Tasa Falsos Positivos (1 - Especificidad)')
ax1.set_ylabel('Tasa Verdaderos Positivos (Recall)')
ax1.set_title('Curva ROC-AUC (Menos útil con desbalance)')
ax1.legend(loc='lower right')

# Curva Precision-Recall
for prob, name in [(y_prob_std, 'Estándar'), (y_prob_w, 'Class Weights'), (y_prob_res, 'SMOTE Resampling')]:
    p, r, _ = precision_recall_curve(y_test, prob)
    ax2.plot(r, p, label=f'{name} (PR-AUC = {auc(r, p):.2f})', lw=2)
ax2.axhline(y=y_test.mean(), color='r', linestyle='--', alpha=0.5, label='Línea Base (Proporción Positiva)')
ax2.set_xlabel('Recall (Sensibilidad)')
ax2.set_ylabel('Precision')
ax2.set_title('Curva Precision-Recall (Obligatoria para clase rara)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("01_curvas_roc_vs_pr.png", dpi=150)
plt.close()
print("\n[INFO] Gráfico '01_curvas_roc_vs_pr.png' guardado en el directorio actual.")


# 4. OPTIMIZACIÓN DE UMBRALES POR FUNCIÓN DE COSTO DE NEGOCIO
print("\n==================================================")
print("4. OPTIMIZACIÓN DE UMBRALES POR FUNCIÓN DE COSTOS (DIAPOSITIVA 19)")
print("==================================================")
# Supuesto de negocio para Riesgo Crediticio:
# - Un Falso Negativo (FN): Otorgar un préstamo a un moroso/fraude. Pérdida financiera severa.
#   Costo FN = $1,000.
# - Un Falso Positivo (FP): Rechazar a un buen pagador. Pérdida administrativa/costo de oportunidad.
#   Costo FP = $100.

cost_fn = 1000.0
cost_fp = 100.0

def calculate_business_cost(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost, tn, fp, fn, tp

# Optimizamos en entrenamiento para evitar sobreajuste de umbral
y_prob_train_w = model_weighted.predict_proba(X_train)[:, 1]
thresholds = np.linspace(0.01, 0.99, 100)
costs = []
for t in thresholds:
    cost, _, _, _, _ = calculate_business_cost(y_train, y_prob_train_w, t)
    costs.append(cost)

best_threshold = thresholds[np.argmin(costs)]
print(f" -> Umbral óptimo calculado en entrenamiento: {best_threshold:.4f}")

# Evaluar rendimiento financiero real en TEST con umbral estándar (0.5) vs umbral óptimo
cost_std, tn_s, fp_s, fn_s, tp_s = calculate_business_cost(y_test, y_prob_w, 0.5)
cost_opt, tn_o, fp_o, fn_o, tp_o = calculate_business_cost(y_test, y_prob_w, best_threshold)

print(f"\nResultados financieros en Test (n={len(y_test)}):")
print(f" * Umbral estándar 0.5:")
print(f"   Matriz: TP={tp_s}, FP={fp_s}, FN={fn_s}, TN={tn_s}")
print(f"   Costo total de decisiones fallidas: ${cost_std:,.2f}")
print(f" * Umbral optimizado de negocio ({best_threshold:.3f}):")
print(f"   Matriz: TP={tp_o}, FP={fp_o}, FN={fn_o}, TN={tn_o}")
print(f"   Costo total de decisiones fallidas: ${cost_opt:,.2f}")
print(f" -> ¡Ahorro financiero total para el negocio!: ${cost_std - cost_opt:,.2f} ({((cost_std - cost_opt)/cost_std)*100:.2f}% de reducción de pérdidas)")

# Gráfico 2: Curva de Costo vs Umbral
plt.figure(figsize=(8, 4.5))
plt.plot(thresholds, costs, 'b-', label='Costo Financiero de Decisiones ($)', lw=2)
plt.axvline(best_threshold, color='green', linestyle='--', label=f'Umbral Óptimo de Negocio ({best_threshold:.2f})', lw=2)
plt.axvline(0.5, color='red', linestyle=':', label='Umbral Estándar Académico (0.5)', lw=2)
plt.xlabel('Umbral de Probabilidad de Riesgo')
plt.ylabel('Costo Financiero de Errores ($)')
plt.title('Curva de Optimización Financiera del Umbral de Decisión')
plt.legend()
plt.tight_layout()
plt.savefig("02_optimizacion_umbral_costos.png", dpi=150)
plt.close()
print("[INFO] Gráfico '02_optimizacion_umbral_costos.png' guardado en el directorio actual.")


# 5. CALIBRACIÓN DE PROBABILIDADES
print("\n==================================================")
print("5. CALIBRACIÓN DE PROBABILIDADES (DIAPOSITIVA 24)")
print("==================================================")

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

# Entrenamos calibradores usando Validación Cruzada sobre el base_pipeline
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Calibrador Platt Scaling (Sigmoide)
cal_sigmoid = CalibratedClassifierCV(estimator=base_pipeline, method='sigmoid', cv=5)
cal_sigmoid.fit(X_train, y_train)

# Calibrador Regresión Isotónica (No paramétrico)
cal_isotonic = CalibratedClassifierCV(estimator=base_pipeline, method='isotonic', cv=5)
cal_isotonic.fit(X_train, y_train)

# Entrenamos el uncalibrated base pipeline para comparar
base_pipeline.fit(X_train, y_train)

# Predicción de probabilidades en test
probs_uncalibrated = base_pipeline.predict_proba(X_test)[:, 1]
probs_sigmoid = cal_sigmoid.predict_proba(X_test)[:, 1]
probs_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]

# Brier Score (MSE de probabilidades predichas contra labels verdaderos)
brier_uncal = brier_score_loss(y_test, probs_uncalibrated)
brier_sig = brier_score_loss(y_test, probs_sigmoid)
brier_iso = brier_score_loss(y_test, probs_isotonic)

print(f"Brier Score Loss en Test (Menor es mejor calibración):")
print(f" - Sin Calibrar (Modelo con pesos): {brier_uncal:.4f}")
print(f" - Calibrado Platt (Sigmoid):        {brier_sig:.4f}")
print(f" - Calibrado Isotónico:               {brier_iso:.4f}")

# Generar reliability curves
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, probs_uncalibrated, n_bins=5)
prob_true_sig, prob_pred_sig = calibration_curve(y_test, probs_sigmoid, n_bins=5)
prob_true_iso, prob_pred_iso = calibration_curve(y_test, probs_isotonic, n_bins=5)

plt.figure(figsize=(8, 5.5))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente Calibrado', alpha=0.5)
plt.plot(prob_pred_uncal, prob_true_uncal, 'r-o', label=f'Sin Calibrar (Brier: {brier_uncal:.3f})', lw=2)
plt.plot(prob_pred_sig, prob_true_sig, 'g-o', label=f'Platt Sigmoid (Brier: {brier_sig:.3f})', lw=2)
plt.plot(prob_pred_iso, prob_true_iso, 'b-o', label=f'Isotónico (Brier: {brier_iso:.3f})', lw=2)
plt.xlabel('Probabilidad Promedio Predicha')
plt.ylabel('Proporción de Positivos Reales')
plt.title('Curva de Calibración / Diagrama de Confiabilidad')
plt.legend()
plt.tight_layout()
plt.savefig("03_calibracion_probabilidades.png", dpi=150)
plt.close()
print("[INFO] Gráfico '03_calibracion_probabilidades.png' guardado en el directorio actual.")

