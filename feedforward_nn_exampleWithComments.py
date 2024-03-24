# This line is used to install the latest version of the imbalanced-learn library, which provides tools for dealing with imbalanced datasets.
# pip install -U imbalanced-learn

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

# Loading the dataset from a CSV file into a pandas DataFrame.
df = pd.read_csv("diabetes.csv")

# Displaying the first few rows of the dataframe to understand its structure and contents.
df.head()

# Displaying the column names of the dataframe to understand the features present.
df.columns

# Plotting histograms for each feature by the outcome (diabetes vs no diabetes).
# This loop iterates over all columns except the last one (assumed to be the 'Outcome' column).
for i in range(len(df.columns[:-1])):
  label = df.columns[i]  # Getting the column name for the current iteration.
  # Plotting the distribution of feature values for samples with diabetes.
  plt.hist(df[df['Outcome']==1][label], color='blue', label="Diabetes", alpha=0.7, density=True, bins=15)
  # Plotting the distribution of feature values for samples without diabetes.
  plt.hist(df[df['Outcome']==0][label], color='red', label="No diabetes", alpha=0.7, density=True, bins=15)
  plt.title(label)  # Setting the title of the histogram to the feature name.
  plt.ylabel("Probability")  # Setting the label for the y-axis.
  plt.xlabel(label)  # Setting the label for the x-axis to the feature name.
  plt.legend()  # Displaying a legend to identify the histograms.
  plt.show()  # Showing the plot.

# Preparing the features (X) and target variable (y) for the model.
X = df[df.columns[:-1]].values  # Extracting feature values as a numpy array.
y = df[df.columns[-1]].values  # Extracting target variable values as a numpy array.

# Checking the shape of the features and target variable arrays to ensure correctness.
X.shape, y.shape

# Initializing a StandardScaler to standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardizing the features.
# Combining the standardized features with the target variable into a single array.
data = np.hstack((X, np.reshape(y, (-1, 1))))
# Creating a new DataFrame from the combined array to facilitate operations.
transformed_df = pd.DataFrame(data, columns=df.columns)

# Initializing a RandomOverSampler to handle class imbalance by oversampling the minority class.
over = RandomOverSampler()
# Resampling the dataset to address the class imbalance.
X, y = over.fit_resample(X, y)
# Combining the resampled features and target variable into a single array for convenience.
data = np.hstack((X, np.reshape(y, (-1, 1))))
# Creating a DataFrame from the resampled dataset for further operations.
transformed_df = pd.DataFrame(data, columns=df.columns)

# Checking the balance of the target classes after resampling.
len(transformed_df[transformed_df["Outcome"]==1]), len(transformed_df[transformed_df["Outcome"]==0])

# Splitting the dataset into training, validation, and test sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Defining a simple neural network model for binary classification.
model = tf.keras.Sequential([
                             tf.keras.layers.Dense(16, activation='relu'),  # First hidden layer with 16 units and ReLU activation.
                             tf.keras.layers.Dense(16, activation='relu'),  # Second hidden layer with 16 units and ReLU activation.
                             tf.keras.layers.Dense(1, activation="sigmoid")  # Output layer with sigmoid activation for binary classification.
])

# Compiling the model with the Adam optimizer, binary crossentropy loss, and tracking accuracy.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Evaluating the model on the training data (for baseline understanding).
model.evaluate(X_train, y_train)

# Evaluating the model on the validation data to see how it performs on unseen data.
model.evaluate(X_valid, y_valid)

#
