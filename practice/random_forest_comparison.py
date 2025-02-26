from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)


models = [model_1, model_2, model_3, model_4, model_5]


# Function for comparing different models
def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


scores = [None]*5


for i in range(0, len(models)):
    mae = score_model(models[i])
    scores[i] = mae
    print("Model %d MAE: %d" % (i+1, mae))

best = models.index(
    min(scores)
)
print(f"Best model is Model {best}")
