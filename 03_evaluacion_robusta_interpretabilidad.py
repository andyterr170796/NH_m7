"""
Módulo 7: Evaluación y Optimización Avanzada
Script 3: Evaluación Robusta, Interpretabilidad y Detección de Drift
------------------------------------------------------------------------------------------------
Este script se enfoca exclusivamente en el Tópico 3 de la agenda de clase.
Enseña cómo llevar a cabo una validación cruzada respetando la dependencia temporal, medir
la importancia de variables global (Permutation Importance) y local (coeficientes), auditar sesgos
por subgrupos (Fairness), evaluar robustez mediante stress testing (ruido/nulos), y simular un monitoreo
de producción continuo para detectar covariate/concept drift usando PSI, KS-test y políticas de reentrenamiento.

Dataset Utilizado: Daily Minimum Temperatures in Melbourne (Datos de Series de Tiempo de Repositorio Público).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de estética de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 14,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})

# 1. CARGA Y PREPROCESAMIENTO DE SERIES DE TIEMPO
print("==================================================")
print("1. CARGANDO Y PREPARANDO DATOS TEMPORALES (SERIE DE TIEMPO)...")
print("==================================================")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
print(" -> Dataset Melbourne Daily Temperatures cargado exitosamente desde GitHub.")

# Renombrar target y crear lags/rolling statistics
df = df.rename(columns={'Temp': 'temperature'})
for lag in [1, 2, 7]:
    df[f'lag_{lag}'] = df['temperature'].shift(lag)

df['rolling_mean_7'] = df['temperature'].shift(1).rolling(window=7).mean()
df['rolling_std_7'] = df['temperature'].shift(1).rolling(window=7).std()

# Eliminar nulos generados
df = df.dropna()

print(f"Rango de datos: {df.index.min().date()} a {df.index.max().date()} ({len(df)} registros).")


# 2. VALIDACIÓN ROBUSTA TEMPORAL
print("\n==================================================")
print("2. VALIDACIÓN ROBUSTA EN SERIES DE TIEMPO (DIAPOSITIVA 25)")
print("==================================================")

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Visualización de los Splits Temporales
plt.figure(figsize=(10, 4.5))
for idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
    plt.plot(train_idx, [idx] * len(train_idx), 'b', lw=6, label='Entrenamiento' if idx == 0 else "")
    plt.plot(val_idx, [idx] * len(val_idx), 'r', lw=6, label='Validación' if idx == 0 else "")

plt.yticks(range(5), [f'Split {i+1}' for i in range(5)])
plt.xlabel('Índice Temporal')
plt.ylabel('Folds de Validación')
plt.title('Esquema de Validación Temporal (TimeSeriesSplit - Expanding Window)')
plt.legend()
plt.tight_layout()
plt.savefig("05_timeseries_splits.png", dpi=150)
plt.close()
print(" -> Gráfico '05_timeseries_splits.png' guardado en el directorio actual.")

# Split en datos históricos (entrenamiento/validación) y datos futuros (producción)
split_date_prod = '1988-01-01'
train_val_df = df[:split_date_prod]
prod_df = df[split_date_prod:].copy()

features = ['lag_1', 'lag_2', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
target = 'temperature'

X_train_val = train_val_df[features]
y_train_val = train_val_df[target]

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_val, y_train_val)

y_pred_val = model.predict(X_train_val)
print(f"Rendimiento Histórico del Modelo (In-Sample MAE): {mean_absolute_error(y_train_val, y_pred_val):.4f}")


# 3. INTERPRETABILIDAD: GLOBAL Y LOCAL
print("\n==================================================")
print("3. EXPLICABILIDAD DEL MODELO (DIAPOSITIVAS 33-37)")
print("==================================================")

# A. Permutation Feature Importance (Global) - Diapositiva 35
from sklearn.inspection import permutation_importance

print("Calculando Permutation Importance...")
perm_importance = permutation_importance(
    model, X_train_val, y_train_val, scoring='neg_mean_absolute_error', n_repeats=5, random_state=42
)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]

print("\nImportancia de Variables (Caída en MAE al permutar):")
for rank, i in enumerate(sorted_idx, 1):
    print(f" {rank}. Variable '{features[i]}': {perm_importance.importances_mean[i]:.4f} ± {perm_importance.importances_std[i]:.4f}")

# B. Coeficientes semánticos / Explicación Local Conceptual - Diapositiva 37
# Para ilustrar una predicción individual de forma local, tomamos un registro de test y analizamos sus desviaciones
sample_idx = 0
sample_x = X_train_val.iloc[[sample_idx]]
sample_y_real = y_train_val.iloc[sample_idx]
sample_y_pred = model.predict(sample_x)[0]

print(f"\nExplicación Local de una Predicción Individual:")
print(f" - Temperatura Real: {sample_y_real}°C")
print(f" - Predicción Modelo: {sample_y_pred:.2f}°C")
print(" Contribución de features (Desviación respecto a la media histórica):")
mean_values = X_train_val.mean()
for feat in features:
    val = sample_x[feat].values[0]
    dev = val - mean_values[feat]
    print(f"  * Variable '{feat}': Valor={val:.2f} (Desviación vs Media Histórica: {dev:+.2f})")

# Graficar interpretabilidad global
plt.figure(figsize=(8, 4.5))
y_pos = np.arange(len(features))
plt.barh(y_pos, perm_importance.importances_mean[sorted_idx][::-1], align='center', color='teal')
plt.yticks(y_pos, np.array(features)[sorted_idx][::-1])
plt.xlabel('Pérdida en Neg-MAE al permutar la variable')
plt.title('Interpretabilidad Global: Permutation Feature Importance')
plt.tight_layout()
plt.savefig("06_global_interpretability.png", dpi=150)
plt.close()
print("\n[INFO] Gráfico '06_global_interpretability.png' guardado en el directorio actual.")


# 4. EVALUACIÓN ROBUSTA POR SUBGRUPOS (FAIRNESS & STABILIDAD)
print("\n==================================================")
print("4. EVALUACIÓN ROBUSTA POR SUBGRUPOS (DIAPOSITIVAS 26 Y 38)")
print("==================================================")
# Auditamos el desempeño del modelo en base a la estación del año (Verano vs Invierno)
# Definimos Invierno (Junio, Julio, Agosto) y Verano (Diciembre, Enero, Febrero) en el histórico
train_val_df_audit = train_val_df.copy()
train_val_df_audit['month'] = train_val_df_audit.index.month
train_val_df_audit['season'] = np.where(train_val_df_audit['month'].isin([6, 7, 8]), 'Invierno',
                                np.where(train_val_df_audit['month'].isin([12, 1, 2]), 'Verano', 'Transición'))

for season in ['Invierno', 'Verano']:
    sub_df = train_val_df_audit[train_val_df_audit['season'] == season]
    sub_x = sub_df[features]
    sub_y = sub_df[target]
    mae_sub = mean_absolute_error(sub_y, model.predict(sub_x))
    print(f" - MAE del Modelo en el subgrupo '{season}': {mae_sub:.4f}°C")


# 5. ROBUSTEZ ANTE DRIFT Y MONITOREO CONTINUO
print("\n==================================================")
print("5. SIMULACIÓN DE DRIFT EN PRODUCCIÓN (DIAPOSITIVA 40 Y 45)")
print("==================================================")

# A. Simulamos COVARIATE DRIFT en Producción (calentamiento físico o falla del sensor: +2.5°C en lags)
prod_df_drift = prod_df.copy()
prod_df_drift['lag_1'] = prod_df_drift['lag_1'] + 2.5
prod_df_drift['lag_2'] = prod_df_drift['lag_2'] + 2.5
prod_df_drift['rolling_mean_7'] = prod_df_drift['rolling_mean_7'] + 2.5

# CÁLCULO DEL POPULATION STABILITY INDEX (PSI) MANUAL
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

psi_lag1 = calculate_psi(X_train_val['lag_1'], prod_df_drift['lag_1'])
print(f"PSI de la variable 'lag_1' (Histórico vs Producción con Drift): {psi_lag1:.4f}")
if psi_lag1 >= 0.25:
    print(" [ALERTA DE RETRAINING]: ¡Drift crítico de variables detectado! (PSI >= 0.25)")

# TEST DE KOLMOGOROV-SMIRNOV (KS)
ks_stat, p_val = stats.ks_2samp(X_train_val['lag_1'], prod_df_drift['lag_1'])
print(f"Test KS para 'lag_1' en Producción:")
print(f" - Estadística KS: {ks_stat:.4f} | p-value: {p_val:.4e}")
if p_val < 0.05:
    print(" -> H0 Rechazada: Las distribuciones en producción son estadísticamente diferentes.")

# B. CONCEPT DRIFT (Cambio estructural en la relación X -> Y: el target real sube permanentemente +4.0°C)
prod_df_concept = prod_df.copy()
prod_df_concept['temperature'] = prod_df_concept['temperature'] + 4.0

# MEDICIÓN DE DEGRADACIÓN Y BUCLE DE RETRAINING MENSUAL
prod_df_concept['year_month'] = prod_df_concept.index.to_period('M')
months = sorted(prod_df_concept['year_month'].unique())

psi_history = []
mae_history = []
active_model = model
trained_on = "Modelo Histórico Inicial"

print("\nIniciando Simulación de Monitoreo Continuo en Producción Batch:")
for m in months:
    batch = prod_df_concept[prod_df_concept['year_month'] == m]
    X_batch = batch[features]
    y_batch = batch['temperature']
    
    # 1. Medir performance
    preds = active_model.predict(X_batch)
    mae = mean_absolute_error(y_batch, preds)
    mae_history.append(mae)
    
    # 2. Medir drift de variables
    psi = calculate_psi(X_train_val['lag_1'], X_batch['lag_1'])
    psi_history.append(psi)
    
    print(f" Lote {m} | MAE = {mae:.3f} | PSI = {psi:.3f} | Estado: {trained_on}")
    
    # Política de Reentrenamiento Automático (MAE > 3.0 o PSI > 0.25)
    if psi >= 0.25 or mae > 3.0:
        print(f"  [DISPARADOR DE ALERTA]: Reentrenando modelo activamente por degradación/drift.")
        current_data = df[:batch.index.max()]
        active_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        active_model.fit(current_data[features], current_data[target])
        trained_on = f"Reentrenado al {batch.index.max().date()}"

# C. STRESS TESTING (DIAPOSITIVA 41)
print("\n==================================================")
print("6. STRESS TESTING Y CONDICIONES ADVERSAS (DIAPOSITIVA 41)")
print("==================================================")

# Simulación de nulos masivos (Sensor caído en un 20% de observaciones)
X_test_missing = prod_df[features].copy()
X_test_missing.loc[X_test_missing.sample(frac=0.2, random_state=42).index, 'lag_1'] = np.nan

print(" -> Test de robustez ante valores nulos en lag_1:")
try:
    _ = model.predict(X_test_missing)
    print("    [EXITOSO]: El modelo pudo procesar nulos nativamente.")
except ValueError as e:
    print("    [LECCIÓN DOCENTE]: Scikit-learn clásico falla ante nulos en inferencia.")
    print("    En producción, el pipeline del estudiante debe integrar un SimpleImputer para contingencias.")

# Inyectamos ruido gaussiano de alta varianza a la variable lag_1
X_test_noisy = prod_df[features].copy()
X_test_noisy['lag_1'] = X_test_noisy['lag_1'] + np.random.normal(0, 10.0, len(X_test_noisy))
mae_normal = mean_absolute_error(prod_df['temperature'], model.predict(prod_df[features]))
mae_noisy = mean_absolute_error(prod_df['temperature'], model.predict(X_test_noisy))
print(f" -> Test ante ruido extremo de sensor:")
print(f"    * MAE Normal: {mae_normal:.4f} | MAE con Ruido Inyectado: {mae_noisy:.4f}")

# Graficar monitoreo en producción
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
x_labels = [str(m) for m in months]

ax1.plot(x_labels, psi_history, 'o-g', lw=2, label='PSI del Lote')
ax1.axhline(0.25, color='r', linestyle='--', label='Drift Crítico (0.25)')
ax1.axhline(0.10, color='orange', linestyle=':', label='Drift Moderado (0.10)')
ax1.set_ylabel('Population Stability Index (PSI)')
ax1.set_title('Monitoreo Continuo: Desviación Distributiva de Datos (PSI)')
ax1.legend()

ax2.plot(x_labels, mae_history, 'o-b', lw=2, label='MAE Lote Mensual')
ax2.axhline(mae_normal, color='k', linestyle=':', label='MAE Óptimo Histórico')
ax2.set_ylabel('Mean Absolute Error (MAE)')
ax2.set_xlabel('Meses de Producción Simulada')
ax2.set_title('Monitoreo Continuo: Degradación de Performance (MAE)')
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("07_production_drift_monitoring.png", dpi=150)
plt.close()
print("\n[INFO] Gráfico '07_production_drift_monitoring.png' guardado en el directorio actual.")

