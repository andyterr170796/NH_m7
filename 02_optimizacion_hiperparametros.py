"""
Módulo 7: Evaluación y Optimización Avanzada
Script 2: Optimización de Hiperparámetros, Estrategias de Búsqueda y Nested CV
------------------------------------------------------------------------------------------------
Este script se enfoca exclusivamente en el Tópico 2 de la agenda de clase.
Enseña cómo tunear hiperparámetros de manera profesional y prevenir el optimismo del score (tuning optimism)
mediante validación cruzada anidada (Nested Cross-Validation), y compara la búsqueda aleatoria con la
búsqueda bayesiana (Optuna).

Dataset Utilizado: SMS Spam Collection (Datos de Texto de Repositorio Público).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estética de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 14,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})

# 1. CARGA Y PREPROCESAMIENTO DE TEXTO (NLP)
print("==================================================")
print("1. CARGANDO DATASET DE TEXTO (SMS SPAM)...")
print("==================================================")

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])
print(" -> Dataset SMS Spam cargado exitosamente desde GitHub.")

# Codificar variable objetivo: 1 = spam (positiva), 0 = ham (normal)
df['target'] = (df['label'] == 'spam').astype(int)

# Separar características (texto) y variable objetivo
X = df['message']
y = df['target']

# Split clásico Train / Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

print(f"Distribución en entrenamiento:")
print(f" - Casos Ham (normales): {len(y_train[y_train == 0])}")
print(f" - Casos Spam (raros):    {len(y_train[y_train == 1])}")


# 2. DEFINICIÓN DEL PIPELINE DE TEXTO Y ESPACIO DE HIPERPARÁMETROS
# Enseña que los hiperparámetros controlan la complejidad del modelo (Regularización, ngram range, use_idf)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=1000)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])


# 3. ESTRATEGIAS DE BÚSQUEDA: RANDOM SEARCH VS OPTIMIZACIÓN BAYESIANA
print("\n==================================================")
print("2. ESTRATEGIAS DE BÚSQUEDA DE HIPERPARÁMETROS (DIAPOSITIVA 29)")
print("==================================================")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score

# Usamos PR-AUC (Average Precision) debido al desbalance
pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')

# Parámetros a explorar
param_dist = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__use_idf': [True, False],
    'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__class_weight': [None, 'balanced']
}

print("A. Ejecutando Búsqueda Aleatoria (Random Search)...")
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring=pr_auc_scorer,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f" -> Mejores parámetros encontrados (Random Search):")
for param, val in random_search.best_params_.items():
    print(f"    * {param}: {val}")
print(f" -> Mejor score PR-AUC interno de CV: {random_search.best_score_:.4f}")

# DEMOSTRACIÓN DE OPTIMIZACIÓN BAYESIANA USANDO OPTUNA
print("\nB. Búsqueda Bayesiana con Optuna (Estándar moderno en la industria)...")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        # Sugerir parámetros bayesianamente
        C = trial.suggest_float('classifier__C', 0.01, 100.0, log=True)
        use_idf = trial.suggest_categorical('vectorizer__use_idf', [True, False])
        ngram = trial.suggest_categorical('vectorizer__ngram_range', [(1, 1), (1, 2)])
        
        test_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english', max_features=1000, use_idf=use_idf, ngram_range=ngram)),
            ('classifier', LogisticRegression(max_iter=1000, C=C, random_state=42))
        ])
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(test_pipeline, X_train, y_train, cv=3, scoring=pr_auc_scorer)
        return scores.mean()
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5) # 5 trials rápidos de ejemplo
    print(" -> ¡Búsqueda Bayesiana (TPE) ejecutada con éxito!")
    print(f"    * Mejores parámetros: {study.best_params}")
    print(f"    * Mejor Score (PR-AUC): {study.best_value:.4f}")
except ImportError:
    print(" -> [AVISO DOCENTE] 'optuna' no está instalado. Para enseñar la Optimización Bayesiana real:")
    print("    pip install optuna")
    print("    Se prosigue sin interrupción offline.")


# 4. PREVINIENDO OPTIMISMO: VALIDACIÓN CRUZADA ANIDADA (NESTED CROSS-VALIDATION)
print("\n==================================================")
print("3. PREVINIENDO OPTIMISMO DE TUNING: NESTED CV (DIAPOSITIVA 31)")
print("==================================================")

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Definimos los bucles de CV:
# - CV Externo: Estima el rendimiento real ante datos nunca vistos.
# - CV Interno: Selecciona la mejor configuración para cada fold externo.
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_optimizer = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=5, # Iteraciones reducidas para velocidad
    cv=inner_cv,
    scoring=pr_auc_scorer,
    random_state=42,
    n_jobs=-1
)

print("Ejecutando Validación Cruzada Anidada (Nested CV)...")
nested_scores = cross_val_score(nested_optimizer, X_train, y_train, cv=outer_cv, scoring=pr_auc_scorer, n_jobs=-1)

print("\nResultados del CV Anidado:")
print(f" - Scores de PR-AUC del bucle externo: {nested_scores}")
print(f" - Rendimiento generalizado real estimado: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print(" -> EXPLICACIÓN PEDAGÓGICA: El mejor score del Randomized Search simple (interno)")
print("    tiende a ser optimista debido a la búsqueda múltiple. El Nested CV nos da una")
print("    estimación real libre de este sesgo de optimismo de tuning.")

# Generamos gráfico para el alumno
plt.figure(figsize=(7, 4.5))
plt.boxplot(nested_scores, vert=False, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))
plt.title('Distribución de Métricas en el Bucle Externo (Nested CV)')
plt.xlabel('PR-AUC (Average Precision)')
plt.yticks([1], ['Nested CV Folds'])
plt.tight_layout()
plt.savefig("04_nested_cv_scores.png", dpi=150)
plt.close()
print("\n[INFO] Gráfico '04_nested_cv_scores.png' guardado en el directorio actual.")
