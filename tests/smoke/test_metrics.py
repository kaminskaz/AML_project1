# Run with: python -m pytest tests.smoke.test_metrics
import numpy as np
from src.metrics.classification import compute_metric, list_supported_metrics

y_true=np.array([0,1,0,1])
y_proba=np.array([0.1,0.9,0.4,0.6])
print('supported_metrics_n=', len(list_supported_metrics()))
for name in ['precision','recall','f1','balanced_accuracy','roc_auc','pr_auc']:
    r=compute_metric(name,y_true,y_proba,threshold=0.5)
    print(name, '->', r.name, r.value)