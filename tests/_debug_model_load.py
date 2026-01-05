from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

p = Path('tmp_if_model.joblib')
df = pd.DataFrame(np.random.RandomState(0).randn(100,3), columns=['f0','f1','f2'])
model = IsolationForest(random_state=0)
model.fit(df)
joblib.dump(model, p)
print('dumped to', p)
loaded = joblib.load(p)
print('loaded type', type(loaded), hasattr(loaded, 'decision_function'), getattr(loaded, 'feature_names_in_', None))

# use our _load_model
from models import predict
m2 = predict._load_model(str(p), force_reload=True)
print('predict._load_model returned', type(m2), hasattr(m2,'decision_function'), getattr(m2, 'feature_names_in_', None))
