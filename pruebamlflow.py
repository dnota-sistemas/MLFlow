# Cargo las librerías
import importlib
import mlflow
importlib.reload(mlflow)
import numpy as np
import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


# Si no existe, creo el experimento
experiment_name = "Rafa"
remote_server_uri = "http://localhost/" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)


if not mlflow.get_experiment_by_name(experiment_name):
    print("No existe")
    mlflow.create_experiment(name=experiment_name) 
else:
    print("Existe")

experiment = mlflow.get_experiment_by_name(experiment_name)

# Setup de MLflow
#mlflow.set_tracking_uri('http://localhost/')

# Cargo los datos
data = load_iris()

# Hago split entre train y test
x_train, x_test, y_train, y_test = train_test_split(
    data['data'],
    data['target'],
    test_size= 0.2,
    random_state= 1234
    )

logged_model = 'runs:/7115959a8cc04f36a67ebcbb80a2d12e/prueba'

# Load model as a PyFuncModel.
rf_class = mlflow.pyfunc.load_model(logged_model)


# Definimos el modelo
#rf_class = RandomForestClassifier()

# Definimos el grid de hiperparámetros
grid = {
    'max_depth':[5,8,10], 
    'min_samples_split':[2,3,4,5],
    'min_samples_leaf':[2,3,4,5],
    'max_features': [2,3]
    }

# Hago el Grid Search
rf_class_grid = GridSearchCV(rf_class, grid, cv = 5) 
rf_class_grid_fit = rf_class_grid.fit(x_train, y_train)

print(f'Best parameters: {rf_class_grid_fit.best_params_}')
print(experiment.experiment_id)


with mlflow.start_run(experiment_id = experiment.experiment_id):

    # Logueo los mejores resultados
    mlflow.log_params(rf_class_grid_fit.best_params_)

    # Obtengo las predicciones
    y_pred = rf_class_grid_fit.predict(x_test)

    # Calculo el acuraccy y el AUC
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}')

    # Log de parámetros
    metrics ={
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall 
        }

    mlflow.log_metrics(metrics)


    # Log model & artifacts
    np.save('x_train', x_train)
    mlflow.log_artifact('x_train.npy')

    mlflow.sklearn.log_model(rf_class_grid_fit, 'prueba')
    # mlflow.sklearn.save_model(rf_class_grid_fit)




    