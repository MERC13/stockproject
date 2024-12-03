'''
Data Caching
Heatmaps
Risk Assessment
LLM to suggest portfolio
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Multiply, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from dataprocessing import dataprocessing
import tensorflow as tf
import joblib
import os
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.interpolate import interp1d





def save_model_and_params(model, params, model_filename, params_filename):
    model.save(model_filename)
    joblib.dump(params, params_filename)





def load_model_and_params(model_filename, params_filename):
    model = load_model(model_filename)
    params = joblib.load(params_filename)
    return model, params






def bayesian_hyperparameter_tuning(x_train, y_train, x_val, y_val):
    # Define the search space
    search_spaces = {
        'model__units': Integer(32, 128),
        'model__learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
        'model__dropout_rate': Real(0.1, 0.5),
        'model__l2_reg': Real(1e-6, 1e-3, prior='log-uniform'),
    }

    # Create a custom scoring function that uses the validation set
    def custom_scorer(estimator, X, y):
        y_pred = estimator.predict(x_val)
        return -mean_squared_error(y_val, y_pred)

    # Create the KerasRegressor
    model = KerasRegressor(
        model=create_model,
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mse'],
        verbose=0
    )

    # Create the BayesSearchCV object
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=50,  # Number of parameter settings that are sampled
        cv=[(slice(None), slice(None))],  # Use all training data
        n_jobs=-1,  # Use all available cores
        verbose=1,
        scoring=custom_scorer,
        random_state=42
    )

    # Fit the BayesSearchCV object to the data
    bayes_search.fit(x_train, y_train)

    # Print the best parameters and score
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best validation MSE: ", -bayes_search.best_score_)

    return bayes_search.best_params_





def create_multi_task_model(units=64, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
    inputs = Input(shape=(shape[0], shape[1]))
    
    # Shared layers
    x = LSTM(units, return_sequences=True, 
             kernel_regularizer=l2(l2_reg), 
             recurrent_regularizer=l2(l2_reg))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg))(x)
    attention = tf.nn.softmax(attention, axis=1)
    context_vector = Multiply()([x, attention])
    x = tf.reduce_sum(context_vector, axis=1)
    
    # Dense layer
    x = Dense(units // 2, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Company-specific output layers
    outputs = {}
    for stock in tech_list:
        safe_name = stock.replace('^', '').replace('.', '_')
        outputs[stock] = Dense(1, name=f'output_{safe_name}', kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=inputs, outputs=list(outputs.values()))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model




def create_model(units=64, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
    inputs = Input(shape=(shape[0], shape[1]))
    
    # LSTM layer with L2 regularization
    x = LSTM(units, return_sequences=True, 
             kernel_regularizer=l2(l2_reg), 
             recurrent_regularizer=l2(l2_reg))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg))(x)
    attention = tf.nn.softmax(attention, axis=1)
    context_vector = Multiply()([x, attention])
    x = tf.reduce_sum(context_vector, axis=1)
    
    # Dense layer
    x = Dense(units // 2, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(1, kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model





def create_or_load_model(MODEL_FILE, PARAMS_FILE, shape, bayesian=False, multi_task=False):
    if os.path.exists(MODEL_FILE) and os.path.exists(PARAMS_FILE):
        print("Loading existing model and parameters...")
        best_model, best_params = load_model_and_params(MODEL_FILE, PARAMS_FILE)
    elif bayesian:
        print("No existing model found. Running Bayesian hyperparameter tuning...")
        x_train_tune, x_val, y_train_tune, y_val = train_test_split(
            x_train[tech_list[0]], y_train[tech_list[0]], test_size=0.2, random_state=42
        )
        best_params = bayesian_hyperparameter_tuning(x_train_tune, y_train_tune, x_val, y_val)
        if multi_task:
            best_model = create_multi_task_model(units=best_params['units'],
                                     learning_rate=best_params['learning_rate'],
                                     dropout_rate=best_params['dropout_rate'],
                                     l2_reg=best_params['l2_reg'])
        else:
            best_model = create_model(units=best_params['model__units'],
                                  learning_rate=best_params['model__learning_rate'],
                                  dropout_rate=best_params['model__dropout_rate'],
                                  l2_reg=best_params['model__l2_reg'])
            
        save_model_and_params(best_model, best_params, MODEL_FILE, PARAMS_FILE)
    else:
        default_params = {
            'units': 64,
            'learning_rate': 0.005,
            'dropout_rate': 0.33,
            'l2_reg': 0.25
        }
        print("No existing model found. Using default parameters...")
        best_params = default_params
        if multi_task:
            best_model = create_multi_task_model(units=best_params['units'],
                                     learning_rate=best_params['learning_rate'],
                                     dropout_rate=best_params['dropout_rate'],
                                     l2_reg=best_params['l2_reg'])
        else:
            best_model = create_model(units=best_params['model__units'],
                                  learning_rate=best_params['model__learning_rate'],
                                  dropout_rate=best_params['model__dropout_rate'],
                                  l2_reg=best_params['model__l2_reg'])
        
        save_model_and_params(best_model, best_params, MODEL_FILE, PARAMS_FILE)
    
    return best_model, best_params





def plot_predictions(y_true, y_pred, name, rmse):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
    plt.title(f'{name} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'static/{name}_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()





def plot_learning_curves(history, stock):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Learning Curves for {stock}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'static/learning_curves_{stock}.png')
    plt.close()
    
    
    
    








# data processing
tech_list = ['^DJI'] #['AAPL', 'GOOG', 'MSFT', 'AMZN']
x_train, y_train, x_test, y_test, shape, scaler = dataprocessing(tech_list)

# model building
model = KerasRegressor(build_fn=create_model, verbose=0)

MODEL_FILE = 'best_stock_model.h5'
PARAMS_FILE = 'best_model_params.joblib'
bayesian = False
multi_task = True

best_model, best_params = create_or_load_model(MODEL_FILE, PARAMS_FILE, shape, bayesian, multi_task)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

for i, stock in enumerate(tech_list):
    history = best_model.fit(x_train[stock], y_train[stock], 
                             batch_size=64, 
                             epochs=100, 
                             validation_data=(x_test[stock], y_test[stock]),
                             callbacks=[early_stop], 
                             verbose=1)
    
    # plotting
    plot_learning_curves(history, stock)

    # Modify these lines
    train_predictions = best_model.predict(x_train[stock])
    test_predictions = best_model.predict(x_test[stock])

    # If the model is multi-task, select the appropriate output
    if multi_task:
        train_predictions = train_predictions[:, i]
        test_predictions = test_predictions[:, i]

    def inverse_transform_data(data, scaler, shape):
        data_feature = np.zeros((len(data), shape[1]))
        data_feature[:, 3] = data.flatten()
        return scaler.inverse_transform(data_feature)[:, 3]

    train_predictions_inv = inverse_transform_data(train_predictions, scaler, shape)
    test_predictions_inv = inverse_transform_data(test_predictions, scaler, shape)
    y_train_inv = inverse_transform_data(y_train[stock], scaler, shape)
    y_test_inv = inverse_transform_data(y_test[stock], scaler, shape)

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predictions_inv))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions_inv))

    plot_predictions(y_train_inv, train_predictions_inv, f"{stock}_train", train_rmse)
    plot_predictions(y_test_inv, test_predictions_inv, f"{stock}_test", test_rmse)

    print(f'Train RMSE for {stock}: {train_rmse:.2f}')
    print(f'Test RMSE for {stock}: {test_rmse:.2f}')

# Save model
best_model.save(MODEL_FILE)