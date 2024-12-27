# -*- coding: utf-8 -*-


# General dependencies
import pandas as pd # for data handling
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # for linear plot
import seaborn as sns # for scatter plot
from sklearn.model_selection import train_test_split #for data preprocessing
import datetime

#%%
# Read sensor data




df = pd.DataFrame({'RefSt': sensor["RefSt"], 'Sensor_O3': sensor["Sensor_O3"], 'Temp': sensor["Temp"], 'RelHum': sensor["RelHum"]})

X = df[['Sensor_O3', 'Temp', 'RelHum']]
Y = df['RefSt']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1, shuffle = False)

df_train = pd.DataFrame({'RefSt': Y_train, 'Sensor_O3': X_train["Sensor_O3"], 'Temp': X_train["Temp"], 'RelHum': X_train["RelHum"]})
df_test = pd.DataFrame({'RefSt': Y_test, 'Sensor_O3': X_test["Sensor_O3"], 'Temp': X_test["Temp"], 'RelHum': X_test["RelHum"]})

from sklearn.metrics import r2_score, mean_absolute_error

def loss_functions(y_true, y_pred):
    print("Loss functions:")
    print("* R-squared =", r2_score(y_true, y_pred))
    print("* MAE =", mean_absolute_error(y_true, y_pred))

def normalize(col):
    μ = col.mean()
    σ = col.std()
    return (col - μ)/σ

df["normRefSt"] = normalize(df["RefSt"])
df["normSensor_O3"] = normalize(df["Sensor_O3"])
df["normTemp"] = normalize(df["Temp"])
df["normRelHum"] = normalize(df["RelHum"])

from keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.optimizers import SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
input_layer = Input(shape=(3,1), dtype='float32')

from sklearn.preprocessing import StandardScaler
# Normalize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## GRU with Attenion with TimeGAN"""

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, GRU, Bidirectional, Input, Attention
from tensorflow.keras.models import Model

# Define the input shape
#input_shape = (3, 1)  # Adjust this based on your data

# Define the model architecture using Functional API
#input_layer = Input(shape=input_shape)
gru1 = GRU(units=100, return_sequences=True)(input_layer)
dropout1 = Dropout(0.2)(gru1)
gru2 = GRU(units=200, return_sequences=True)(dropout1)
gru3 = GRU(units=100, return_sequences=True)(gru2)
dropout2 = Dropout(0.2)(gru3)

# Use Keras built-in Attention layer
# Attention requires two inputs (query, value), so use the same output as both.
attention_output = Attention()([dropout2, dropout2])  # Query and Value are the same

# Output layer
dense_output = Dense(1, activation='linear')(attention_output)

# Create the model
model = Model(inputs=input_layer, outputs=dense_output)

# Compile the model
model.compile(loss='mean_absolute_error', optimizer='adam')

# Train the model
atgruhistory = model.fit(X_train, Y_train, epochs=100, batch_size=32)

import matplotlib.pyplot as plt

# Assuming you already have the training history object from model.fit()
# This will contain the loss values for each epoch
history = atgruhistory  # The object returned by model.fit()

# Get the loss values from the history
loss_values = history.history['loss']

# Plot the loss values
plt.figure(figsize=(6,4))
plt.plot(loss_values, label='Training Loss', color='b')

# Add labels and title
#plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.savefig("TimeGANGRUAttention_lossTimeGAN.png", dpi=300)
plt.show()

import os
# Save the original model in HDF5 format
model.save("model.h5")

# Get the size of the original model
original_model_size = os.path.getsize("model.h5") / 1024  # Size in KB
print(f"Original GRU-Attention Time GAN model size = {original_model_size:.2f} KB")

X_test.shape

df_test.shape



# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# 
# # Predict the target values for the test set
# Y_pred = model.predict(X_test)
# 
# # Option 1: Use predictions from the last timestep
# Y_pred_flat = Y_pred[:, -1, 0]  # Take the last timestep for each sample
# 
# # Option 2: Average predictions across timesteps
# Y_pred_flat = Y_pred.mean(axis=1).squeeze()  # Average across timesteps
# 
# # Compute MAE, MSE, and R²
# mae = mean_absolute_error(Y_test, Y_pred_flat)
# mse = mean_squared_error(Y_test, Y_pred_flat)
# r2 = r2_score(Y_test, Y_pred_flat)
# rmse_avg = np.sqrt(mse)
# # Print the evaluation metrics
# print("Mean Absolute Error (MAE):", mae)
# print("Mean Squared Error (MSE):", mse)
# print("RMSE Score:", rmse_avg)
# 
#

print("Shape of Y_test:", Y_test.shape)
print("Shape of Y_pred:", Y_pred.shape)

df_test[ "TimeGANGRUAtt"]= Y_pred_flat
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANGRUAtt"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+GRUAttention Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANGRUAttention_predOriginal.png", dpi=300)
plt.show()



"""## GRU-Attention Lite GRU with Attenion with TimeGAN"""

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable resource variables to handle RNNs
converter.experimental_enable_resource_variables = True

# Enable Select TensorFlow Ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Prevent lowering of TensorList ops
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_float_model = converter.convert()

# Save the model
with open("model_float.tflite", "wb") as f:
    f.write(tflite_float_model)

# Check the size of the TFLite model
float_model_size = len(tflite_float_model) / 1024  # KBs
print(f"Float model size = {float_model_size:.2f}KB")

import tensorflow as tf

# Load the TFLite float model
interpreter = tf.lite.Interpreter(model_path="model_float.tflite")

# Allocate tensors (required before inference)
interpreter.allocate_tensors()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print details to understand input/output shape and data types
print("Input Details:", input_details)
print("Output Details:", output_details)

import numpy as np
import tensorflow as tf

# Assuming `X_test` is the array with shape (1249, 3)
input_data = X_test  # Use your test data here

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_float.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check expected shape and dtype
expected_shape = input_details[0]['shape']  # Example: [1, 3, 1]
expected_dtype = input_details[0]['dtype']  # Should be tf.float32

print("Expected Input Shape:", expected_shape)
print("Expected Input Dtype:", expected_dtype)

# Reshape input_data to match expected shape
sample_index = 0  # Use the first sample for prediction
reshaped_input_data = np.reshape(input_data[sample_index], (1, 3, 1))  # Reshape to [1, 3, 1]

# Ensure data type is FLOAT32
reshaped_input_data = reshaped_input_data.astype(np.float32)  # Explicit conversion

print("Prepared Input Shape:", reshaped_input_data.shape)
print("Prepared Input Dtype:", reshaped_input_data.dtype)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], reshaped_input_data)

# Run inference
interpreter.invoke()

# Retrieve predictions
predictions = interpreter.get_tensor(output_details[0]['index'])
print("Predictions:", predictions)



import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predictions from the TFLite model (assuming predictions is already available)
# Example: predictions = [[...]]
pred_flattened = predictions.reshape(-1)  # Flatten predictions to 1D array

# Corresponding Y_test values
y_test_sample = Y_test[:len(pred_flattened)]  # Match the number of samples

# Compute metrics
mse = mean_squared_error(y_test_sample, pred_flattened)
mae = mean_absolute_error(y_test_sample, pred_flattened)
r2 = r2_score(y_test_sample, pred_flattened)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")



#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Method 1: Use the last timestep's prediction for each sample
Y_pred_last_timestep = all_predictions[:, -1]  # Last timestep prediction for each sample

# Method 2: Average predictions across all timesteps for each sample
Y_pred_avg = all_predictions.mean(axis=1).squeeze()  # Average prediction across timesteps

# Compute metrics for Method 1 (Last Timestep)
mae_last = mean_absolute_error(Y_test, Y_pred_last_timestep)
mse_last = mean_squared_error(Y_test, Y_pred_last_timestep)
rmse_last = np.sqrt(mse_last)
r2_last = r2_score(Y_test, Y_pred_last_timestep)

# Compute metrics for Method 2 (Average Timestep)
mae_avg = mean_absolute_error(Y_test, Y_pred_avg)
mse_avg = mean_squared_error(Y_test, Y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
r2_avg = r2_score(Y_test, Y_pred_avg)

# Print the evaluation metrics for both methods
print("Method 1: Last Timestep Predictions")
print(f"MAE: {mae_last:.4f}, MSE: {mse_last:.4f}, RMSE: {rmse_last:.4f}, R²: {r2_last:.4f}")

print("\nMethod 2: Average Timestep Predictions")
print(f"MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}, R²: {r2_avg:.4f}")

# Plot predictions vs actual values for both methods
plt.figure(figsize=(12, 6))

# Plot for Method 1 (Last Timestep)
plt.subplot(1, 2, 1)
plt.plot(Y_test, label='Actual Values', color='blue')
plt.plot(Y_pred_last_timestep, label='Predicted Values (Last Timestep)', color='orange', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Last Timestep Predictions vs Actual Values')
plt.legend()
plt.grid(True)

# Plot for Method 2 (Average Timestep)
plt.subplot(1, 2, 2)
plt.plot(Y_test, label='Actual Values', color='blue')
plt.plot(Y_pred_avg, label='Predicted Values (Average Timestep)', color='green', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Average Timestep Predictions vs Actual Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

df_test[ "TimeGANGRUAttLite"]= Y_pred_last_timestep
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANGRUAttLite"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+Light GRUAttention Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANGRUAttentionLite_predictionsLITE.png", dpi=300)
plt.show()

"""## GRUAttention Quantized Model"""

# Optionally, apply quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Get the size of the quantized model
quantized_model_size = len(tflite_quantized_model) / 1024  # Size in KB
print(f"Quantized model size = {quantized_model_size:.2f} KB")
print(f"Quantized model is {quantized_model_size * 100 / float_model_size:.2f}% of the float model size.")

# Save the quantized model
with open("gru_model_quantized.tflite", "wb") as f:
    f.write(tflite_quantized_model)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import numpy as np
# 
# # Load the TFLite float model
# interpreter_float = tf.lite.Interpreter(model_path="model_float.tflite")
# interpreter_float.allocate_tensors()
# 
# # Get input and output details for the float model
# input_details_float = interpreter_float.get_input_details()
# output_details_float = interpreter_float.get_output_details()
# 
# # Load the TFLite quantized model
# interpreter_quantized = tf.lite.Interpreter(model_path="gru_model_quantized.tflite")
# interpreter_quantized.allocate_tensors()
# 
# # Get input and output details for the quantized model
# input_details_quantized = interpreter_quantized.get_input_details()
# output_details_quantized = interpreter_quantized.get_output_details()
# 
# # Example: Prepare your input data
# input_data = X_test  # Assuming X_test is your test data
# 
# # Function to run inference and return predictions
# def get_predictions(interpreter, input_data, input_details, output_details):
#     predictions = []
#     for i in range(input_data.shape[0]):
#         reshaped_input_data = np.reshape(input_data[i], (1, 3, 1))  # Adjust shape as needed
#         reshaped_input_data = reshaped_input_data.astype(np.float32)  # Ensure dtype is float32
# 
#         # Set the input tensor
#         interpreter.set_tensor(input_details[0]['index'], reshaped_input_data)
# 
#         # Run inference
#         interpreter.invoke()
# 
#         # Retrieve predictions and append them
#         prediction = interpreter.get_tensor(output_details[0]['index'])
#         predictions.append(prediction)
# 
#     return np.array(predictions).reshape(input_data.shape[0], -1)  # Flatten predictions
# 
# # Get predictions for the float model
# predictions_float = get_predictions(interpreter_float, input_data, input_details_float, output_details_float)
# 
# # Get predictions for the quantized model
# predictions_quantized = get_predictions(interpreter_quantized, input_data, input_details_quantized, output_details_quantized)
# 
# # Print predictions (you can compare float vs quantized predictions)
# print("Predictions (Float model):", predictions_float)
# print("Predictions (Quantized model):", predictions_quantized)
#

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming all_predictions is a 2D array where each row corresponds to a test sample,
# and each column is a prediction for a specific timestep (shape: [num_samples, timesteps])

# Method 1: Use the last timestep's prediction for each sample
Y_pred_last_timestep = all_predictions[:, -1]  # Get the prediction from the last timestep

# Method 2: Average predictions across all timesteps for each sample
Y_pred_avg = all_predictions.mean(axis=1).squeeze()  # Compute the average prediction across timesteps

# Ensure the ground truth and predictions are aligned in terms of shape
print(f"Shape of Y_test: {Y_test.shape}")
print(f"Shape of Y_pred_last_timestep: {Y_pred_last_timestep.shape}")
print(f"Shape of Y_pred_avg: {Y_pred_avg.shape}")

# Compute metrics for Method 1 (Last Timestep)
mae_last = mean_absolute_error(Y_test, Y_pred_last_timestep)
mse_last = mean_squared_error(Y_test, Y_pred_last_timestep)
rmse_last = np.sqrt(mse_last)
r2_last = r2_score(Y_test, Y_pred_last_timestep)

# Compute metrics for Method 2 (Average Timestep)
mae_avg = mean_absolute_error(Y_test, Y_pred_avg)
mse_avg = mean_squared_error(Y_test, Y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
r2_avg = r2_score(Y_test, Y_pred_avg)

# Print the evaluation metrics for both methods
print("Method 1: Last Timestep Predictions")
print(f"MAE: {mae_last:.4f}, MSE: {mse_last:.4f}, RMSE: {rmse_last:.4f}, R²: {r2_last:.4f}")

print("\nMethod 2: Average Timestep Predictions")
print(f"MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}, R²: {r2_avg:.4f}")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Method 1: Use the last timestep's prediction for each sample
# Y_pred_last_timestep = all_predictions[:, -1]  # Get the prediction from the last timestep
#

df_test[ "TimeGANGRUAttQNT"]= Y_pred_last_timestep
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANGRUAttQNT"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+Quantized GRUAttention Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANGRUAttentionQuantized_pred.png", dpi=300)
plt.show()

"""## Comparison of all GRU models"""

import matplotlib.pyplot as plt
import numpy as np

# Data for performance metrics
models = ['TimeGAN+GRUAttention', 'TimeGAN+Float GRUAttention', 'TimeGAN+Quantized GRUAttention']
mse = [0.2952, 0.2952, 0.3212]
mae = [0.4171, 0.4171, 0.4375]
rmse = [0.5433, 0.5430, 0.5667]

# Data for model sizes (in KB)
model_sizes = [3595.93, 1210.79, 356.45]

# Set the positions for the bars
x = np.arange(len(models))  # the label locations

# Create and save figure for Performance Metrics (MSE, MAE, RMSE)
fig1, ax1 = plt.subplots(figsize=(7, 5))

# Define hatch patterns
hatch_patterns = ['/', '\\', 'x']  # Different hatch styles for different models

# Plot for MSE, MAE, RMSE
ax1.bar(x - 0.2, mse, 0.2, label='MSE', color='skyblue', hatch=hatch_patterns[0])
ax1.bar(x, mae, 0.2, label='MAE', color='lightcoral', hatch=hatch_patterns[1])
ax1.bar(x + 0.2, rmse, 0.2, label='RMSE', color='lightgreen', hatch=hatch_patterns[2])

# Labeling for the first plot
ax1.set_xlabel('Models')
ax1.set_ylabel('Values')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8)
ax1.legend()

# Removing grid and title as per request
ax1.grid(False)  # Remove grid

# Save the first figure with dpi=300
fig1.tight_layout()
fig1.savefig('/content/performance_metrics_comparison_GRUAttention.png', dpi=300)

# Create and save figure for Model Size Comparison
fig2, ax2 = plt.subplots(figsize=(7, 5))

# Plot for model sizes
ax2.bar(x, model_sizes, color='lightblue', hatch='//')

# Labeling for the second plot
ax2.set_xlabel('Models')
ax2.set_ylabel('Model Size (KB)')
ax2.set_xticks(x)
ax2.set_xticklabels(models,fontsize=8)

# Removing grid and title as per request
ax2.grid(False)  # Remove grid
# Save the second figure with dpi=300
fig2.tight_layout()
fig2.savefig('/content/model_size_comparison_GRUAttention.png', dpi=300)

# Optionally, you can display the plots
plt.show()

import os
import zipfile
from google.colab import files

# Specify the directory and the extensions you want to download
directory = '/content/'
extensions = ['.png','csv']  # Add more extensions as needed

# List all files in the directory
files_in_directory = os.listdir(directory)

# Filter files with the specified extensions
files_to_download = [file for file in files_in_directory if any(file.endswith(ext) for ext in extensions)]

# Define the name of the zip file
zip_filename = '/content/GRULite.zip'

# Create a zip file and add the matching files to it
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files_to_download:
        file_path = os.path.join(directory, file)
        zipf.write(file_path, os.path.basename(file))  # Add file to zip

# Download the zip file
files.download(zip_filename)

"""## Large Models Transformer

# New section
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization

# If you haven't already, make sure you have the necessary libraries
# pip install tensorflow numpy matplotlib scikit-learn
# Define the input shape (assuming time series data with 3 time steps and 1 feature)
input_layer = Input(shape=(3, 1), dtype='float32')

# Transformer Encoder Block (Multi-Head Attention + Feed-Forward)
# Attention Layer
attention = MultiHeadAttention(num_heads=2, key_dim=32)(input_layer, input_layer)
attention = LayerNormalization()(attention)
attention = Dropout(0.1)(attention)

# Feed Forward Layer
ffn = Dense(64, activation='relu')(attention)
ffn = Dropout(0.1)(ffn)
ffn = Dense(1)(ffn)

# Create the model
transformer_model = Model(inputs=input_layer, outputs=ffn)

# Compile the model
transformer_model.compile(loss='mean_absolute_error', optimizer='adam')

# Model summary
transformer_model.summary()

# Assuming X_train and Y_train are your training data
# X_train should have shape (num_samples, 3, 1)
# Y_train should have shape (num_samples, 1)

historytranformer = transformer_model.fit(X_train, Y_train, epochs=500, batch_size=32)

# Save the model after training (if needed)
transformer_model.save('transformer_model.h5')

import os


# Get the size of the original model
original_model_size = os.path.getsize("transformer_model.h5") / 1024  # Size in KB
print(f"Original TimeGAN+Transformer model size = {original_model_size:.2f} KB")

import matplotlib.pyplot as plt

# Assuming you already have the training history object from model.fit()
# This will contain the loss values for each epoch
history = historytranformer  # The object returned by model.fit()

# Get the loss values from the history
loss_values = history.history['loss']

# Plot the loss values
plt.figure(figsize=(6,4))
plt.plot(loss_values, label='Training Loss', color='b')

# Add labels and title
#plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.savefig("TimeGANTransformer.png", dpi=300)
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# 
# # Predict the target values for the test set
# Y_pred = transformer_model.predict(X_test)
# 
# # Option 1: Use predictions from the last timestep
# _pred_flat = Y_pred[:, -1, 0]  # Take the last timestep for each sample
# 
# # Option 2: Average predictions across timesteps
# Y_pred_flat = Y_pred.mean(axis=1).squeeze()  # Average across timesteps
# 
# # Compute MAE, MSE, and R²
# mae = mean_absolute_error(Y_test, Y_pred_flat)
# mse = mean_squared_error(Y_test, Y_pred_flat)
# r2 = r2_score(Y_test, Y_pred_flat)
# rmse_avg = np.sqrt(mse)
# # Print the evaluation metrics
# print("Mean Absolute Error (MAE):", mae)
# print("Mean Squared Error (MSE):", mse)
# print("RMSE Score:", rmse_avg)
# 
#

df_test[ "TimeGANTransformer"]= Y_pred_flat
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANTransformer"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+ TransformerPredictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANGRUAttention_predOriginal.png", dpi=300)
plt.show()

"""## Transformer Lite"""

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(transformer_model)

# Enable resource variables to handle RNNs
converter.experimental_enable_resource_variables = True

# Enable Select TensorFlow Ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Prevent lowering of TensorList ops
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_float_model = converter.convert()

# Save the model
with open("transformermodel_float.tflite", "wb") as f:
    f.write(tflite_float_model)

# Check the size of the TFLite model
float_model_size = len(tflite_float_model) / 1024  # KBs
print(f"Float model size = {float_model_size:.2f}KB")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import numpy as np
# import tensorflow as tf
# 
# # Assuming `X_test` is the array with shape (1249, 3)
# input_data = X_test  # Use your test data here
# 
# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="transformermodel_float.tflite")
# interpreter.allocate_tensors()
# 
# # Get input details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# 
# # Check expected shape and dtype
# expected_shape = input_details[0]['shape']  # Example: [1, 3, 1]
# expected_dtype = input_details[0]['dtype']  # Should be tf.float32
# 
# print("Expected Input Shape:", expected_shape)
# print("Expected Input Dtype:", expected_dtype)
# 
# # Reshape input_data to match expected shape
# sample_index = 0  # Use the first sample for prediction
# reshaped_input_data = np.reshape(input_data[sample_index], (1, 3, 1))  # Reshape to [1, 3, 1]
# 
# # Ensure data type is FLOAT32
# reshaped_input_data = reshaped_input_data.astype(np.float32)  # Explicit conversion
# 
# print("Prepared Input Shape:", reshaped_input_data.shape)
# print("Prepared Input Dtype:", reshaped_input_data.dtype)
# 
# # Set the input tensor
# interpreter.set_tensor(input_details[0]['index'], reshaped_input_data)
# 
# # Run inference
# interpreter.invoke()
# 
# # Retrieve predictions
# predictions = interpreter.get_tensor(output_details[0]['index'])
# print("Predictions:", predictions)
#

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predictions from the TFLite model (assuming predictions is already available)
# Example: predictions = [[...]]
pred_flattened = predictions.reshape(-1)  # Flatten predictions to 1D array

# Corresponding Y_test values
y_test_sample = Y_test[:len(pred_flattened)]  # Match the number of samples

# Compute metrics
mse = mean_squared_error(y_test_sample, pred_flattened)
mae = mean_absolute_error(y_test_sample, pred_flattened)
rmse= np.sqrt(mse)
r2 = r2_score(y_test_sample, pred_flattened)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Method 1: Use the last timestep's prediction for each sample
Y_pred_last_timestep = all_predictions[:, -1]  # Last timestep prediction for each sample

# Method 2: Average predictions across all timesteps for each sample
Y_pred_avg = all_predictions.mean(axis=1).squeeze()  # Average prediction across timesteps

# Compute metrics for Method 1 (Last Timestep)
mae_last = mean_absolute_error(Y_test, Y_pred_last_timestep)
mse_last = mean_squared_error(Y_test, Y_pred_last_timestep)
rmse_last = np.sqrt(mse_last)
r2_last = r2_score(Y_test, Y_pred_last_timestep)

# Compute metrics for Method 2 (Average Timestep)
mae_avg = mean_absolute_error(Y_test, Y_pred_avg)
mse_avg = mean_squared_error(Y_test, Y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
r2_avg = r2_score(Y_test, Y_pred_avg)

# Print the evaluation metrics for both methods
print("Method 1: Last Timestep Predictions")
print(f"MAE: {mae_last:.4f}, MSE: {mse_last:.4f}, RMSE: {rmse_last:.4f}, R²: {r2_last:.4f}")

print("\nMethod 2: Average Timestep Predictions")
print(f"MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}, R²: {r2_avg:.4f}")

# Plot predictions vs actual values for both methods
plt.figure(figsize=(12, 6))

# Plot for Method 1 (Last Timestep)
plt.subplot(1, 2, 1)
plt.plot(Y_test, label='Actual Values', color='blue')
plt.plot(Y_pred_last_timestep, label='Predicted Values (Last Timestep)', color='orange', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Last Timestep Predictions vs Actual Values')
plt.legend()
plt.grid(True)

# Plot for Method 2 (Average Timestep)
plt.subplot(1, 2, 2)
plt.plot(Y_test, label='Actual Values', color='blue')
plt.plot(Y_pred_avg, label='Predicted Values (Average Timestep)', color='green', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Average Timestep Predictions vs Actual Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

df_test[ "TimeGANTransLite"]= Y_pred_last_timestep
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANTransLite"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+Light Transformer Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANTransformerLite_predictionsLITE.png", dpi=300)
plt.show()

"""## Quantized Transformer

# New section
"""

# Optionally, apply quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Get the size of the quantized model
quantized_model_size = len(tflite_quantized_model) / 1024  # Size in KB
print(f"Quantized model size = {quantized_model_size:.2f} KB")
print(f"Quantized model is {quantized_model_size * 100 / float_model_size:.2f}% of the float model size.")

# Save the quantized model
with open("transformer_model_quantized.tflite", "wb") as f:
    f.write(tflite_quantized_model)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import numpy as np
# 
# # Load the TFLite float model
# interpreter_float = tf.lite.Interpreter(model_path="transformermodel_float.tflite")
# interpreter_float.allocate_tensors()
# 
# # Get input and output details for the float model
# input_details_float = interpreter_float.get_input_details()
# output_details_float = interpreter_float.get_output_details()
# 
# # Load the TFLite quantized model
# interpreter_quantized = tf.lite.Interpreter(model_path="transformer_model_quantized.tflite")
# interpreter_quantized.allocate_tensors()
# 
# # Get input and output details for the quantized model
# input_details_quantized = interpreter_quantized.get_input_details()
# output_details_quantized = interpreter_quantized.get_output_details()
# 
# # Example: Prepare your input data
# input_data = X_test  # Assuming X_test is your test data
# 
# # Function to run inference and return predictions
# def get_predictions(interpreter, input_data, input_details, output_details):
#     predictions = []
#     for i in range(input_data.shape[0]):
#         reshaped_input_data = np.reshape(input_data[i], (1, 3, 1))  # Adjust shape as needed
#         reshaped_input_data = reshaped_input_data.astype(np.float32)  # Ensure dtype is float32
# 
#         # Set the input tensor
#         interpreter.set_tensor(input_details[0]['index'], reshaped_input_data)
# 
#         # Run inference
#         interpreter.invoke()
# 
#         # Retrieve predictions and append them
#         prediction = interpreter.get_tensor(output_details[0]['index'])
#         predictions.append(prediction)
# 
#     return np.array(predictions).reshape(input_data.shape[0], -1)  # Flatten predictions
# 
# # Get predictions for the float model
# predictions_float = get_predictions(interpreter_float, input_data, input_details_float, output_details_float)
# 
# # Get predictions for the quantized model
# predictions_quantized = get_predictions(interpreter_quantized, input_data, input_details_quantized, output_details_quantized)
# 
# # Print predictions (you can compare float vs quantized predictions)
# print("Predictions (Float model):", predictions_float)
# print("Predictions (Quantized model):", predictions_quantized)
#

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming all_predictions is a 2D array where each row corresponds to a test sample,
# and each column is a prediction for a specific timestep (shape: [num_samples, timesteps])

# Method 1: Use the last timestep's prediction for each sample
Y_pred_last_timestep = all_predictions[:, -1]  # Get the prediction from the last timestep

# Method 2: Average predictions across all timesteps for each sample
Y_pred_avg = all_predictions.mean(axis=1).squeeze()  # Compute the average prediction across timesteps

# Ensure the ground truth and predictions are aligned in terms of shape
print(f"Shape of Y_test: {Y_test.shape}")
print(f"Shape of Y_pred_last_timestep: {Y_pred_last_timestep.shape}")
print(f"Shape of Y_pred_avg: {Y_pred_avg.shape}")

# Compute metrics for Method 1 (Last Timestep)
mae_last = mean_absolute_error(Y_test, Y_pred_last_timestep)
mse_last = mean_squared_error(Y_test, Y_pred_last_timestep)
rmse_last = np.sqrt(mse_last)
r2_last = r2_score(Y_test, Y_pred_last_timestep)

# Compute metrics for Method 2 (Average Timestep)
mae_avg = mean_absolute_error(Y_test, Y_pred_avg)
mse_avg = mean_squared_error(Y_test, Y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
r2_avg = r2_score(Y_test, Y_pred_avg)

# Print the evaluation metrics for both methods
print("Method 1: Last Timestep Predictions")
print(f"MAE: {mae_last:.4f}, MSE: {mse_last:.4f}, RMSE: {rmse_last:.4f}, R²: {r2_last:.4f}")

print("\nMethod 2: Average Timestep Predictions")
print(f"MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}, R²: {r2_avg:.4f}")

df_test[ "TimeGANTransQ"]= Y_pred_last_timestep
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANTransQ"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+Quantized Transformer Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANTransformernQuantized_pred.png", dpi=300)
plt.show()

"""## Comparison of Transformer Models."""

import matplotlib.pyplot as plt
import numpy as np

# Data for inference times (in milliseconds)
models = [
    'TimeGAN+GRUAttention',
    'TimeGAN+Light GRUAttention',
    'TimeGAN+Quantized GRUAttention',
    'TimeGAN+Transformer',
    'TimeGAN+Light Transformer',
    'TimeGAN+Quantized Transformer',
    'TimeGAN+Temporal Fusion Transformer',
    'TimeGAN+Float Temporal Fusion Transformer',
    'TimeGAN+Quantized Temporal Fusion Transformer'
]

# Inference times (in milliseconds)
inference_times = [
    1340,  # TimeGAN+GRUAttention (1.34 s)
    647,   # TimeGAN+Light GRUAttention (647 ms)
    19.1,  # TimeGAN+Quantized GRUAttention (19.1 µs) -> 0.0191 ms
    235,   # TimeGAN+Transformer (235 ms)
    24,    # TimeGAN+Light Transformer (24 ms)
    125,   # TimeGAN+Quantized Transformer (125 ms)
    384,   # TimeGAN+Temporal Fusion Transformer (384 ms)
    23.4,  # TimeGAN+Float Temporal Fusion Transformer (23.4 ms)
    178    # TimeGAN+Quantized Temporal Fusion Transformer (178 ms)
]

# Set the positions for the bars, ensuring bars for the same group are next to each other
x = np.arange(0, len(models), 3)  # Create an array with positions for all models, spaced for groups

# Bar width
bar_width = 0.2

# Create and save figure for Inference Time Comparison
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Define hatch patterns for different models
hatch_patterns = ['/', '\\', 'x', '-', '|', 'O', '*', '+', '.']  # Different hatch styles for different models

# Plot for Inference Times
for i, model in enumerate(models):
    ax1.bar(x[i % 3] + (i // 3) * bar_width, inference_times[i], bar_width, color='skyblue', hatch=hatch_patterns[i % len(hatch_patterns)])

# Labeling for the first plot
ax1.set_xlabel('Models')
ax1.set_ylabel('Inference Time (ms)')

# Set x-ticks at the center of each group of bars
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(['Model'] * len(x), fontsize=8)  # Display 'Model' on the x-axis

# Create the legend with model names
ax1.legend(models, loc='upper left', fontsize=8, bbox_to_anchor=(1, 1))

# Removing grid and title as per request
ax1.grid(False)  # Remove grid

# Save the first figure with dpi=300
fig1.tight_layout()
fig1.savefig('inference_time_comparison_ms.png', dpi=300)

# Optionally, you can display the plot
plt.show()

import os
import zipfile
from google.colab import files

# Specify the directory and the extensions you want to download
directory = '/content/'
extensions = ['.png']  # Add more extensions as needed

# List all files in the directory
files_in_directory = os.listdir(directory)

# Filter files with the specified extensions
files_to_download = [file for file in files_in_directory if any(file.endswith(ext) for ext in extensions)]

# Define the name of the zip file
zip_filename = '/content/TransformerLite.zip'

# Create a zip file and add the matching files to it
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files_to_download:
        file_path = os.path.join(directory, file)
        zipf.write(file_path, os.path.basename(file))  # Add file to zip

# Download the zip file
files.download(zip_filename)

"""## Temporal Transformer Model TFT"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention

# Input shape for time series data
input_layer = Input(shape=(3, 1), dtype='float32')  # 3 time steps, 1 feature

# ----- Temporal Fusion Layers -----

# 1. Variable Selection Network (VSN)
def variable_selection(input_layer):
    # Apply a dense layer to learn feature importance
    vsn = Dense(32, activation='relu')(input_layer)
    vsn = LayerNormalization()(vsn)
    return vsn

# 2. Gated Residual Network (GRN)
def gated_residual_network(input_layer, units=64):
    grn = Dense(units, activation='relu')(input_layer)
    grn = Dropout(0.2)(grn)
    grn = Dense(units)(grn)
    grn = LayerNormalization()(grn)
    return grn

# Apply the Variable Selection Network
vsn_output = variable_selection(input_layer)

# Apply the Gated Residual Network
grn_output = gated_residual_network(vsn_output)

# ----- Attention Mechanism -----
# Multi-Head Attention
attention = MultiHeadAttention(num_heads=2, key_dim=32)(grn_output, grn_output)
attention = LayerNormalization()(attention)
attention = Dropout(0.1)(attention)

# ----- Temporal Fusion -----
# Add LSTM layer after attention
lstm_layer = LSTM(64, return_sequences=False)(attention)
lstm_layer = Dropout(0.2)(lstm_layer)

# ----- Output Layer -----
# Feed-Forward Network (FFN) for regression task
ffn = Dense(64, activation='relu')(lstm_layer)
ffn = Dropout(0.2)(ffn)
output_layer = Dense(1)(ffn)

# Define the Temporal Fusion Transformer model
tft_model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model with Mean Absolute Error loss function
tft_model.compile(loss='mean_absolute_error', optimizer='adam')

# Model Summary
tft_model.summary()


# Train the model
history = tft_model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt

# Assuming you already have the training history object from model.fit()
# This will contain the loss values for each epoch
  # The object returned by model.fit()

# Get the loss values from the history
loss_values = history.history['loss']

# Plot the loss values
plt.figure(figsize=(6,4))
plt.plot(loss_values, label='Training Loss', color='b')

# Add labels and title
#plt.title('Training Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.savefig("TimeGANTTFTLoss.png", dpi=300)
plt.show()

# Assuming X_train and Y_train are your training data


# Save the model after training (if needed)
tft_model.save('TFT_transformer_model.h5')
import os


# Get the size of the original model
original_model_size = os.path.getsize("TFT_transformer_model.h5") / 1024  # Size in KB
print(f"Original TimeGAN+TFT model size = {original_model_size:.2f} KB")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# 
# # Predict the target values for the test set
# Y_pred = tft_model.predict(X_test)
# 
# # Option 1: Use predictions from the last timestep (not applicable for 2D output)
# # _pred_flat = Y_pred[:, -1, 0]  # This line is not needed since Y_pred is 2D
# 
# # Option 2: Average predictions across timesteps
# Y_pred_flat = Y_pred.squeeze()  # Since Y_pred is 2D, we can just flatten it
# 
# # Compute MAE, MSE, and R²
# mae = mean_absolute_error(Y_test, Y_pred_flat)
# mse = mean_squared_error(Y_test, Y_pred_flat)
# r2 = r2_score(Y_test, Y_pred_flat)
# rmse_avg = np.sqrt(mse)
# 
# # Print the evaluation metrics
# print("TFT Mean Absolute Error (MAE):", mae)
# print("TFT Mean Squared Error (MSE):", mse)
# print("TFT RMSE Score:", rmse_avg)
# print("TFT R2:", r2)
#

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import time
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# 
# # Record the start time
# start_time = time.time()
# 
# # Predict the target values for the test set
# Y_pred = tft_model.predict(X_test)
# 
# # Option 1: Use predictions from the last timestep (not applicable for 2D output)
# # _pred_flat = Y_pred[:, -1, 0]  # This line is not needed since Y_pred is 2D
# 
# # Option 2: Average predictions across timesteps
# Y_pred_flat = Y_pred.squeeze()  # Since Y_pred is 2D, we can just flatten it
# 
# 
# 
# # Record the end time and compute the inference time
# end_time = time.time()
# inference_time = end_time - start_time
# 
# # Print the inference time
# print(f"Inference time: {inference_time:.4f} seconds")
#

df_test[ "TimeGANTFT"]= Y_pred_flat
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANTFT"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+ Temporal Fusion Transformer Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANTFT_predOriginal.png", dpi=300)
plt.show()

"""## TFT Lite"""

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(tft_model)

# Enable resource variables to handle RNNs
converter.experimental_enable_resource_variables = True

# Enable Select TensorFlow Ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Prevent lowering of TensorList ops
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_float_model = converter.convert()

# Save the model
with open("tft_model_float.tflite", "wb") as f:
    f.write(tflite_float_model)

# Check the size of the TFLite model
float_model_size = len(tflite_float_model) / 1024  # KBs
print(f"Float model size = {float_model_size:.2f}KB")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import numpy as np
# import tensorflow as tf
# 
# # Assuming `X_test` is the array with shape (1249, 3)
# input_data = X_test  # Use your test data here
# 
# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path="tft_model_float.tflite")
# interpreter.allocate_tensors()
# 
# # Get input details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# 
# # Check expected shape and dtype
# expected_shape = input_details[0]['shape']  # Example: [1, 3, 1]
# expected_dtype = input_details[0]['dtype']  # Should be tf.float32
# 
# print("Expected Input Shape:", expected_shape)
# print("Expected Input Dtype:", expected_dtype)
# 
# # Reshape input_data to match expected shape
# sample_index = 0  # Use the first sample for prediction
# reshaped_input_data = np.reshape(input_data[sample_index], (1, 3, 1))  # Reshape to [1, 3, 1]
# 
# # Ensure data type is FLOAT32
# reshaped_input_data = reshaped_input_data.astype(np.float32)  # Explicit conversion
# 
# print("Prepared Input Shape:", reshaped_input_data.shape)
# print("Prepared Input Dtype:", reshaped_input_data.dtype)
# 
# # Set the input tensor
# interpreter.set_tensor(input_details[0]['index'], reshaped_input_data)
# 
# # Run inference
# interpreter.invoke()
# 
# # Retrieve predictions
# predictions = interpreter.get_tensor(output_details[0]['index'])
# print("tft_model_float.tflit Predictions:", predictions)
#

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predictions from the TFLite model (assuming predictions is already available)
# Example: predictions = [[...]]
pred_flattened = predictions.reshape(-1)  # Flatten predictions to 1D array

# Corresponding Y_test values
y_test_sample = Y_test[:len(pred_flattened)]  # Match the number of samples

# Compute metrics
mse = mean_squared_error(y_test_sample, pred_flattened)
mae = mean_absolute_error(y_test_sample, pred_flattened)
rmse= np.sqrt(mse)
r2 = r2_score(y_test_sample, pred_flattened)

print(f"TFT Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Method 1: Use the last timestep's prediction for each sample
Y_pred_last_timestep = all_predictions[:, -1]  # Last timestep prediction for each sample

# Method 2: Average predictions across all timesteps for each sample
Y_pred_avg = all_predictions.mean(axis=1).squeeze()  # Average prediction across timesteps

# Compute metrics for Method 1 (Last Timestep)
mae_last = mean_absolute_error(Y_test, Y_pred_last_timestep)
mse_last = mean_squared_error(Y_test, Y_pred_last_timestep)
rmse_last = np.sqrt(mse_last)
r2_last = r2_score(Y_test, Y_pred_last_timestep)

# Compute metrics for Method 2 (Average Timestep)
mae_avg = mean_absolute_error(Y_test, Y_pred_avg)
mse_avg = mean_squared_error(Y_test, Y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
r2_avg = r2_score(Y_test, Y_pred_avg)

# Print the evaluation metrics for both methods
print("Method 1: Last Timestep Predictions")
print(f"MAE: {mae_last:.4f}, MSE: {mse_last:.4f}, RMSE: {rmse_last:.4f}, R²: {r2_last:.4f}")

print("\nMethod 2: Average Timestep Predictions")
print(f"MAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}, R²: {r2_avg:.4f}")

# Plot predictions vs actual values for both methods
plt.figure(figsize=(12, 6))

# Plot for Method 1 (Last Timestep)
plt.subplot(1, 2, 1)
plt.plot(Y_test, label='Actual Values', color='blue')
plt.plot(Y_pred_last_timestep, label='Predicted Values (Last Timestep)', color='orange', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Last Timestep Predictions vs Actual Values')
plt.legend()
plt.grid(True)

# Plot for Method 2 (Average Timestep)
plt.subplot(1, 2, 2)
plt.plot(Y_test, label='Actual Values', color='blue')
plt.plot(Y_pred_avg, label='Predicted Values (Average Timestep)', color='green', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.title('Average Timestep Predictions vs Actual Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

df_test[ "TimeGANTFTfloat"]= Y_pred_last_timestep
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANTFTfloat"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+Light Temporal Fusion Transformer Predictions"], fontsize=8, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANTFTLite_predictionsLITE.png", dpi=300)
plt.show()

"""## Quantized TFT"""

# Optionally, apply quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Get the size of the quantized model
quantized_model_size = len(tflite_quantized_model) / 1024  # Size in KB
print(f"Quantized model size = {quantized_model_size:.2f} KB")
print(f"Quantized model is {quantized_model_size * 100 / float_model_size:.2f}% of the float model size.")

# Save the quantized model
with open("tft_model_float_quantized.tflite", "wb") as f:
    f.write(tflite_quantized_model)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import numpy as np
# 
# 
# # Get input and output details for the float model
# input_details_float = interpreter_float.get_input_details()
# output_details_float = interpreter_float.get_output_details()
# 
# # Load the TFLite quantized model
# interpreter_quantized = tf.lite.Interpreter(model_path="tft_model_float_quantized.tflite")
# interpreter_quantized.allocate_tensors()
# 
# # Get input and output details for the quantized model
# input_details_quantized = interpreter_quantized.get_input_details()
# output_details_quantized = interpreter_quantized.get_output_details()
# 
# # Example: Prepare your input data
# input_data = X_test  # Assuming X_test is your test data
# 
# # Function to run inference and return predictions
# def get_predictions(interpreter, input_data, input_details, output_details):
#     predictions = []
#     for i in range(input_data.shape[0]):
#         reshaped_input_data = np.reshape(input_data[i], (1, 3, 1))  # Adjust shape as needed
#         reshaped_input_data = reshaped_input_data.astype(np.float32)  # Ensure dtype is float32
# 
#         # Set the input tensor
#         interpreter.set_tensor(input_details[0]['index'], reshaped_input_data)
# 
#         # Run inference
#         interpreter.invoke()
# 
#         # Retrieve predictions and append them
#         prediction = interpreter.get_tensor(output_details[0]['index'])
#         predictions.append(prediction)
# 
#     return np.array(predictions).reshape(input_data.shape[0], -1)  # Flatten predictions
# 
# # Get predictions for the float model
# predictions_float = get_predictions(interpreter_float, input_data, input_details_float, output_details_float)
# 
# # Get predictions for the quantized model
# predictions_quantized = get_predictions(interpreter_quantized, input_data, input_details_quantized, output_details_quantized)
# 
# # Print predictions (you can compare float vs quantized predictions)
# print("Predictions (Float model):", predictions_float)
# print("Predictions (Quantized model):", predictions_quantized)
#

# Commented out IPython magic to ensure Python compatibility.
#  %%time
 #Predictions for the quantized model
predictions_quantized = get_predictions(interpreter_quantized, input_data, input_details_quantized, output_details_quantized)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming all_predictions is a 2D array where each row corresponds to a test sample,
# and each column is a prediction for a specific timestep (shape: [num_samples, timesteps])

# Method 1: Use the last timestep's prediction for each sample
Y_pred_last_timestep = all_predictions[:, -1]  # Get the prediction from the last timestep

# Method 2: Average predictions across all timesteps for each sample
Y_pred_avg = all_predictions.mean(axis=1).squeeze()  # Compute the average prediction across timesteps

# Ensure the ground truth and predictions are aligned in terms of shape
print(f"Shape of Y_test: {Y_test.shape}")
print(f"Shape of Y_pred_last_timestep: {Y_pred_last_timestep.shape}")
print(f"Shape of Y_pred_avg: {Y_pred_avg.shape}")

# Compute metrics for Method 1 (Last Timestep)
mae_last = mean_absolute_error(Y_test, Y_pred_last_timestep)
mse_last = mean_squared_error(Y_test, Y_pred_last_timestep)
rmse_last = np.sqrt(mse_last)
r2_last = r2_score(Y_test, Y_pred_last_timestep)

# Compute metrics for Method 2 (Average Timestep)
mae_avg = mean_absolute_error(Y_test, Y_pred_avg)
mse_avg = mean_squared_error(Y_test, Y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
r2_avg = r2_score(Y_test, Y_pred_avg)

# Print the evaluation metrics for both methods
print("Method 1: Last Timestep Predictions")
print(f"MAE: {mae_last:.4f}, MSE: {mse_last:.4f}, RMSE: {rmse_last:.4f}, R²: {r2_last:.4f}")

print("\nMethod 2: Average Timestep Predictions")
print(f"TFTQMAE: {mae_avg:.4f}, MSE: {mse_avg:.4f}, RMSE: {rmse_avg:.4f}, R²: {r2_avg:.4f}")

df_test[ "TimeGANTFTQ"]= Y_pred_last_timestep
# Plotting RefSt and KNN_Pred

df_test[["RefSt", "TimeGANTFTQ"]].plot(figsize=(8, 6), linewidth=2)

# Set title and labels
#plt.title("KNN Predictions vs Reference Station", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Values", fontsize=12)
# Remove grid
plt.grid(False)
# Add legend
plt.legend(["Reference Station", "TimeGAN+Quantized Temporal Fusion Transformer Predictions"], fontsize=10, loc="best")
plt.xticks(rotation = 20)
# Rotate x-axis ticks for better readability
plt.xticks(rotation=20, fontsize=10)
# Display the plot
plt.tight_layout()
plt.savefig("TimeGANTFTnQuantized_pred.png", dpi=300)
plt.show()

"""## Comparing all TFT models"""

import matplotlib.pyplot as plt
import numpy as np

# Data for performance metrics (Updated for TimeGAN+TFT models)
models = ['TimeGAN+Temporal FT', 'TimeGAN+Float Temporal FT', 'TimeGAN+Quantized Temporal FT']
mse = [2.4776, 1.7434, 0.3212]  # MSE values for each model
mae = [1.5049, 1.3204, 0.4375]  # MAE values for each model
rmse = [1.5740, 1.3204, 0.5667]  # RMSE values for each model

# Data for model sizes (in KB)
model_sizes = [800.04, 259.83, 97.34]

# Set the positions for the bars
x = np.arange(len(models))  # the label locations

# Create and save figure for Performance Metrics (MSE, MAE, RMSE)
fig1, ax1 = plt.subplots(figsize=(7, 5))

# Define hatch patterns for different models
hatch_patterns = ['/', '\\', 'x']  # Different hatch styles for different models

# Plot for MSE, MAE, RMSE
ax1.bar(x - 0.2, mse, 0.2, label='MSE', color='skyblue', hatch=hatch_patterns[0])
ax1.bar(x, mae, 0.2, label='MAE', color='lightcoral', hatch=hatch_patterns[1])
ax1.bar(x + 0.2, rmse, 0.2, label='RMSE', color='lightgreen', hatch=hatch_patterns[2])

# Labeling for the first plot
ax1.set_xlabel('Models')
ax1.set_ylabel('Values')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8)
ax1.legend()

# Removing grid and title as per request
ax1.grid(False)  # Remove grid

# Save the first figure with dpi=300
fig1.tight_layout()
fig1.savefig('performance_metrics_comparison_TFT.png', dpi=300)

# Create and save figure for Model Size Comparison
fig2, ax2 = plt.subplots(figsize=(7, 5))

# Plot for model sizes
ax2.bar(x, model_sizes, color='lightblue', hatch='//')

# Labeling for the second plot
ax2.set_xlabel('Models')
ax2.set_ylabel('Model Size (KB)')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8)

# Removing grid and title as per request
ax2.grid(False)  # Remove grid

# Save the second figure with dpi=300
fig2.tight_layout()
fig2.savefig('model_size_comparison_TFT.png', dpi=300)

# Optionally, you can display the plots
plt.show()

"""Comparing all TFT models Times"""

import matplotlib.pyplot as plt
import numpy as np

# Data for inference times (in seconds or milliseconds)
models = ['TimeGAN+GRUAttention', 'TimeGAN+Light GRUAttention', 'TimeGAN+Quantized GRUAttention',
          'TimeGAN+Transformer', 'TimeGAN+Light Transformer', 'TimeGAN+Quantized Transformer',
          'TimeGAN+Temporal Fusion Transformer', 'TimeGAN+Float Temporal Fusion Transformer',
          'TimeGAN+Quantized Temporal Fusion Transformer']

inference_times = [1.34, 0.647, 0.0000191, 0.235, 0.024, 0.125, 0.384, 0.0234, 0.178]  # Inference times in seconds

# Set the positions for the bars
x = np.arange(len(models))  # the label locations

# Create and save figure for Inference Time Comparison
fig1, ax1 = plt.subplots(figsize=(7, 5))

# Define hatch patterns
hatch_patterns = ['/', '\\', 'x', '-', '|', 'O', '*', '+', '.']  # Different hatch styles for different models

# Plot for Inference Times
ax1.bar(x - 0.2, inference_times, 0.2, label='Inference Time', color='skyblue', hatch=hatch_patterns[0])

# Labeling for the first plot
ax1.set_xlabel('Models')
ax1.set_ylabel('Inference Time (seconds)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8, rotation=45, ha="right")
ax1.legend()

# Removing grid and title as per request
ax1.grid(False)  # Remove grid

# Save the first figure with dpi=300
fig1.tight_layout()
fig1.savefig('/content/inference_time_comparison.png', dpi=300)

# Optionally, you can display the plot
plt.show()

import os
import zipfile
from google.colab import files

# Specify the directory and the extensions you want to download
directory = '/content/'
extensions = ['.png']  # Add more extensions as needed

# List all files in the directory
files_in_directory = os.listdir(directory)

# Filter files with the specified extensions
files_to_download = [file for file in files_in_directory if any(file.endswith(ext) for ext in extensions)]

# Define the name of the zip file
zip_filename = '/content/TFTLite.zip'

# Create a zip file and add the matching files to it
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files_to_download:
        file_path = os.path.join(directory, file)
        zipf.write(file_path, os.path.basename(file))  # Add file to zip

# Download the zip file
files.download(zip_filename)