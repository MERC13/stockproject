'''
Dynamic Charts: Allow users to interact with stock price charts, zoom in on specific time periods,
and view detailed data points.

Heatmaps: Use heatmaps to show market trends and highlight significant changes in stock prices.

Risk Assessment: Provide tools for users to assess the risk of their portfolios and suggest ways to
diversify and mitigate risk.

Performance Tracking: Allow users to track the performance of their portfolios over time and compare
it to market benchmarks.

LLM to suggest portfolio

anomaly detection
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Multiply, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from dataprocessing import dataprocessing
import tensorflow as tf
from scipy.stats import randint, uniform
import joblib
import os






def save_model_and_params(model, params, model_filename, params_filename):
    model.save(model_filename)
    joblib.dump(params, params_filename)





def load_model_and_params(model_filename, params_filename):
    model = load_model(model_filename)
    params = joblib.load(params_filename)
    return model, params





def time_series_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mape_scores = []
    mae_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    return np.mean(mape_scores), np.mean(mae_scores)





def create_model(units=64, learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
    inputs = Input(shape=(shape[0], shape[1]))
    
    # First LSTM layer with L2 regularization
    x = Bidirectional(LSTM(units, return_sequences=True, 
                           kernel_regularizer=l2(l2_reg), 
                           recurrent_regularizer=l2(l2_reg)))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second LSTM layer
    x = Bidirectional(LSTM(units // 2, return_sequences=True, 
                           kernel_regularizer=l2(l2_reg), 
                           recurrent_regularizer=l2(l2_reg)))(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Attention mechanism
    attention = Dense(units // 2, activation='tanh', kernel_regularizer=l2(l2_reg))(x)
    attention = Dense(1, activation='softmax', kernel_regularizer=l2(l2_reg))(attention)
    context_vector = Multiply()([x, attention])
    x = tf.reduce_sum(context_vector, axis=1)
    
    # Output layer
    outputs = Dense(1, kernel_regularizer=l2(l2_reg))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error')
    return model






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
tech_list = ['AAPL']  # 'GOOG', 'MSFT', 'AMZN'
x_train, y_train, x_test, y_test, shape, scaler = dataprocessing(tech_list)






# model building
model = KerasRegressor(build_fn=create_model, verbose=0)

MODEL_FILE = 'best_stock_model.h5'
PARAMS_FILE = 'best_model_params.joblib'

if os.path.exists(MODEL_FILE) and os.path.exists(PARAMS_FILE):
    print("Loading existing model and parameters...")
    best_model, best_params = load_model_and_params(MODEL_FILE, PARAMS_FILE)
else:
    print("No existing model found. Running RandomizedSearchCV...")
    param_distributions = {
        'units': randint(32, 256),
        'learning_rate': uniform(0.0001, 0.1),
        'epochs': randint(50, 300),
        'dropout_rate': uniform(0.1, 0.5),
        'l2_reg': uniform(0.0001, 0.1)
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
                                       n_iter=20, cv=TimeSeriesSplit(n_splits=5), 
                                       verbose=2, n_jobs=-1, scoring='neg_mean_absolute_error')
    
    random_search_result = random_search.fit(x_train[tech_list[0]], y_train[tech_list[0]])

    print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))

    best_model = create_model(units=random_search_result.best_params_['units'],
                              learning_rate=random_search_result.best_params_['learning_rate'],
                              dropout_rate=random_search_result.best_params_['dropout_rate'],
                              l2_reg=random_search_result.best_params_['l2_reg'])

    save_model_and_params(best_model, random_search_result.best_params_, MODEL_FILE, PARAMS_FILE)
    best_params = random_search_result.best_params_

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

for stock in tech_list:
    history = best_model.fit(x_train[stock], y_train[stock], 
                             batch_size=64, 
                             epochs=best_params['epochs'], 
                             validation_data=(x_test[stock], y_test[stock]),  # Use test data for validation
                             callbacks=[early_stop], 
                             verbose=1)
    
    
    
    
    
    
    # plotting
    plot_learning_curves(history, stock)

    train_predictions = best_model.predict(x_train[stock])
    test_predictions = best_model.predict(x_test[stock])

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