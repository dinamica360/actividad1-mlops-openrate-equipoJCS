# Experimentos de Logistic Regression (tuning de C y class_weight)

Se ejecutaron tres experimentos adicionales de regresión logística para analizar el efecto de la regularización y del balanceo de clases en la predicción de `target_opened`.

## Configuración de experimentos

- E2: LogisticRegression(C=0.1, class_weight=None)
- E3: LogisticRegression(C=10.0, class_weight=None)
- E4: LogisticRegression(C=1.0, class_weight=balanced)

## Resultados

### E2 — C=0.1, class_weight=None

- Accuracy: 0.8950
- Precision: 0.8914
- Recall: 0.9873
- F1-score: 0.9369
- ROC AUC: 0.9592

### E3 — C=10.0, class_weight=None

- Accuracy: 0.9000
- Precision: 0.9059
- Recall: 0.9747
- F1-score: 0.9390
- ROC AUC: 0.9581

### E4 — C=1.0, class_weight=balanced

- Accuracy: 0.9100
- Precision: 0.9605
- Recall: 0.9241
- F1-score: 0.9419
- ROC AUC: 0.9592

