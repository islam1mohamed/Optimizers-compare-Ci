#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import tensorflow as tf
import time


# In[3]:


import pickle

# Path to the saved file
input_file = "/kaggle/input/casia-dataset-pkl/casia_webface_part_minv.pkl"

# Load the data
with open(input_file, "rb") as f:
    loaded_data = pickle.load(f)

# Access the loaded images and labels
images = loaded_data["images"]
labels = loaded_data["labels"]


# In[4]:


images = np.array(images)
labels = np.array(labels)

print(images.shape)
print(labels.shape)
# print(set(labels))
#no 28


# ### split

# #### normalize float32

# In[5]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Normalize images in batches and ensure float32
def normalize_images_in_batches(images, batch_size=50):
    normalized_images = np.zeros_like(images, dtype=np.float32)
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        mean = np.mean(batch, axis=(1, 2), keepdims=True)  # Compute mean per image
        std = np.std(batch, axis=(1, 2), keepdims=True)    # Compute std per image
        normalized_images[i:i + batch_size] = (batch - mean) / (std + 1e-8)
        # if i%500==0:
        #     print(i)
    return normalized_images.astype(np.float32)  # Ensure float32



# Normalize images
normalized_images = normalize_images_in_batches(images, batch_size=100)


# In[6]:


import matplotlib.pyplot as plt

def visualize_original_and_normalized(original_images, normalized_images, num_images=5):
    """
        original_images (np.ndarray): Array of original images.
        normalized_images (np.ndarray): Array of normalized images.
        num_images (int): Number of images to visualize.
    """
    plt.figure(figsize=(10, 4 * num_images))

    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(original_images[i])
        plt.title(f"Original Image {i+1}")
        plt.axis('off')

        # Normalized image (rescaled for display)
        normalized_rescaled = (normalized_images[i] - normalized_images[i].min()) / (
            normalized_images[i].max() - normalized_images[i].min()
        )
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(normalized_rescaled)
        plt.title(f"Normalized Image {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function
visualize_original_and_normalized(images, normalized_images, num_images=3)


# In[7]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Assuming `images` and `labels` are the dataset in uint8
# Keep data in uint8 format
print(f"Original Data Type: {normalized_images.dtype}, Shape: {normalized_images.shape}")

# Split the data
train_images, temp_images, train_labels, temp_labels = train_test_split(
    normalized_images, labels, test_size=0.3, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)


# ## clear unused 

# In[8]:


images = []
temp_images = []
labels = []


# In[9]:


# assert train_images.shape == train_labels.shape
# assert val_images.shape == val_labels.shape
# assert test_images.shape == test_labels.shape

print(f"Train Images Shape: {train_images.shape}")
print(f"Validation Images Shape: {val_images.shape}")
print(f"Test Images Shape: {test_images.shape}")



# ## ready 32

# #### define data for tensor

# In[10]:


# Prepare TensorFlow datasets
batch_size = 16
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_images, train_images))
    .batch(batch_size)
    .shuffle(1000)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val_images, val_images))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((test_images, test_images))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)


# # trainer + Model

# ## Vanilla model

# In[11]:


#  return tf.keras.Model(inputs, outputs, name="Vanilla_Autoencoder")
import numpy as np
import tensorflow as tf

import random
import os

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix hash-based randomness
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    tf.random.set_seed(seed)  # TensorFlow
    
    # Optional: GPU determinism (may slightly reduce performance)
    tf.config.experimental.enable_op_determinism()
    
# Set seed before training
set_seed(42)

def vanilla_autoencoder_lessnodes(input_shape):

    # Encoder
    print(f"Input shape: {input_shape}")
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    latent_space = tf.keras.layers.Dense(32, activation='relu', name="latent_space")(x)

    # Decoder
    x = tf.keras.layers.Dense(32, activation='relu')(latent_space)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(int(np.prod(input_shape)), activation='sigmoid')(x)
    
    outputs = tf.keras.layers.Reshape(input_shape)(x)  # Match input shape
    
    # Full autoencoder model
    autoencoder = tf.keras.Model(inputs, outputs, name="Vanilla_Autoencoder")
    # Encoder model (for feature extraction)
    encoder = tf.keras.Model(inputs, latent_space, name="Encoder")

    return autoencoder, encoder
def vanilla_autoencoder_longer(input_shape):

    # Encoder
    print(f"Input shape: {input_shape}")
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Flatten()(inputs)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    latent_space = tf.keras.layers.Dense(64, activation='relu', name="latent_space")(x)

    # Decoder
    x = tf.keras.layers.Dense(64, activation='relu')(latent_space)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(int(np.prod(input_shape)), activation='sigmoid')(x)
    
    outputs = tf.keras.layers.Reshape(input_shape)(x)  # Match input shape
    
        # Full autoencoder model
    autoencoder = tf.keras.Model(inputs, outputs, name="Vanilla_Autoencoder")

    # Encoder model (for feature extraction)
    encoder = tf.keras.Model(inputs, latent_space, name="Encoder")

    return autoencoder, encoder



def vanilla_autoencoder(input_shape):

    # Encoder
    print(f"Input shape: {input_shape}")
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Flatten()(inputs)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    latent_space = tf.keras.layers.Dense(64, activation='relu', name="latent_space")(x)

    # Decoder
    x = tf.keras.layers.Dense(64, activation='relu')(latent_space)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(int(np.prod(input_shape)), activation='sigmoid')(x)
    
    outputs = tf.keras.layers.Reshape(input_shape)(x)  # Match input shape
    
    # Full autoencoder model
    autoencoder = tf.keras.Model(inputs, outputs, name="Vanilla_Autoencoder")

    # Encoder model (for feature extraction)
    encoder = tf.keras.Model(inputs, latent_space, name="Encoder")

    return autoencoder, encoder


# autoencoder.summary()



# In[12]:


set_seed(42)
# Instantiate and compile the autoencoder
input_shape = (112, 112, 3)  # Update if needed
print("less nodes")
autoencoder , encoder = vanilla_autoencoder_lessnodes(input_shape=input_shape)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # mean square error

start_time = time.time()
# Train the autoencoder
epochs = 10
history = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = autoencoder.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
# set_seed(42)

                                            ##########
## pt2
# Instantiate and compile the autoencoder
print("longer")
autoencoder , encoder = vanilla_autoencoder_longer(input_shape=input_shape)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # mean square error

start_time = time.time()
# Train the autoencoder

history = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = autoencoder.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

                                    ##########
## pt3
# set_seed(42)
print("original")
# Instantiate and compile the autoencoder
autoencoder , encoder = vanilla_autoencoder(input_shape=input_shape)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # mean square error

start_time = time.time()
# Train the autoencoder

history = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = autoencoder.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")



# Define the file path
log_file = '/kaggle/working/training_logs/training_history.txt'

# Ensure the directory exists
import os
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Append the history to the file
with open(log_file, 'a') as f:
    f.write(f"Training Session Log\nvanilla\n")
    f.write(f"Epochs: {len(history.history['loss'])}\n")
    f.write(f"Training Loss: {history.history['loss']}\n")
    f.write(f"Validation Loss: {history.history['val_loss']}\n")
    f.write(f"time: {time.time()-start_time}\n")
    # f.write("-" * 50 + "\n")
print(f"Training history appended to {log_file}!")




# ## Convolutional_model

# In[13]:


import tensorflow as tf
# conv takes number of filters over images as input argument & the more filter used the smaller the image becomes
def convolutional_autoencoder(input_shape):

    # Encoder
    print(f"Input shape: {input_shape}")
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional Encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Latent Space
    latent_space = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="latent_space")(x)

    # Convolutional Decoder
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(latent_space)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    # Output Layer
    outputs = tf.keras.layers.Conv2DTranspose(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
   
    autoencoder = tf.keras.Model(inputs, outputs, name="Vanilla_Autoencoder")

    encoder = tf.keras.Model(inputs, latent_space, name="Encoder")

    return autoencoder, encoder




# In[14]:


print("original")
# Instantiate and compile the autoencoder
input_shape = (112,112,3)
conv_autoencoder , conv_encoder = convolutional_autoencoder(input_shape=input_shape)
conv_autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # mean square error

start_time = time.time()
# Train the autoencoder
epochs = 10
conv_history = conv_autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = conv_autoencoder.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# Define the file path
log_file = '/kaggle/working/training_logs/training_history.txt'
# Ensure the directory exists
import os
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Append the history to the file
with open(log_file, 'a') as f:
    f.write(f"Training for conolution\n")
    f.write(f"Epochs: {len(conv_history.history['loss'])}\n")
    f.write(f"Training Loss: {conv_history.history['loss']}\n")
    f.write(f"Validation Loss: {conv_history.history['val_loss']}\n")
    f.write(f"time: {time.time()-start_time}\n")
    f.write("-" * 25 + "\n")
print(f"Training history appended to {log_file}!")


# ## variational model
# 
# 

# ### saving models

# In[15]:


## Save model

# Save the entire model
autoencoder.save("/kaggle/working/autoencoder_model.h5")

conv_autoencoder.save("/kaggle/working/conv_model.h5")
# platues after 3rd run at around 69 loss afterward only 1 loss decrease


# ## load model if exist and train

# In[16]:


# from tensorflow.keras.models import load_model

# # Load the model
# autoencoder = load_model('vanilla_autoencoder.h5')
# print("Model loaded successfully!")

# # Optionally, resume training
# history = autoencoder.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=epochs,  # Continue training for more epochs
#     verbose=1
# # )

# from tensorflow.keras.models import load_model

# # Load the full model
# full_model = load_model("/kaggle/working/vae_full_model.h5")

# # Load the encoder
# encoder = load_model("/kaggle/working/vae_encoder.h5")



# ### Show loss of Vanilla

# In[17]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# ### show loss conv

# In[18]:


import matplotlib.pyplot as plt
print("conv loss")
plt.plot(conv_history.history['loss'], label='Training Loss')
plt.plot(conv_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# ### Variational draw

# ## Calculate accuracy with classifier

# In[19]:


import numpy as np

def calculate_accuracy(predicted_images, normalized_images, labels):
    """
    Calculate accuracy by matching predicted images to normalized images and checking labels.

        predicted_images (np.ndarray): Array of predicted/reconstructed images.
        normalized_images (np.ndarray): Array of normalized input images.
        labels (np.ndarray): Array of true labels corresponding to normalized images.

    """
    # assert predicted_images.shape == normalized_images.shape, "Shape mismatch between predicted and normalized images."
    # assert len(labels) == len(normalized_images), "Mismatch between labels and normalized images."
    
    # Initialize counters
    idx= []
    correct_matches = 0
    total_images = len(predicted_images)

    # Loop through each predicted image
    for i, predicted in enumerate(predicted_images):
        if i%200==0:
            print(i)
        # Calculate the pixel-wise mean squared error with all normalized images
        errors = np.mean((normalized_images - predicted) ** 2, axis=(1, 2, 3))
        
        # Find the closest normalized image
        closest_index = np.argmin(errors)
        
        # If the closest match is below the threshold, check the labels
        if labels[closest_index] == labels[i]:
            correct_matches += 1
            idx.append(i)
            

    # Calculate accuracy
    accuracy = (correct_matches / total_images) * 100
    return accuracy, idx

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def calculate_accuracy_euc_knn(train_images, test_images, train_labels, test_labels, k=4):

    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Train the k-NN classifier on the training data
    knn.fit(train_images, train_labels)

    # Predict the labels for the test images
    predictions = knn.predict(test_images)

    # Identify correctly classified indices
    correct_indices = [i for i, (pred, true) in enumerate(zip(predictions, test_labels)) if pred == true]

    # Calculate accuracy
    accuracy = (len(correct_indices) / len(test_labels)) * 100

    return accuracy, correct_indices


# ## for Vanilla

# In[20]:


# Predict normalized images using the trained autoencoder
train_images_latent = encoder.predict(train_images, batch_size=32, verbose=1)
test_images_latent = encoder.predict(test_images, batch_size=32, verbose=1)

# Verify the shape of the predicted images
print(f"Predicted Images Shape: {train_images_latent.shape}")
print(f"Predicted Images Shape: {test_images_latent.shape}")


# In[21]:


# print(normalized_images.shape, predicted_images.shape)
# normalized_images = normalized_images.astype(np.uint8)
# predicted_images = predicted_images.astype(np.uint8)
# difference = train_images - predicted_images
# print(difference.shape)  # Should be (10000, 112, 112, 3)


# In[22]:


time1 = time.time()
acc,idx = calculate_accuracy_euc_knn(train_images_latent, test_images_latent,train_labels, test_labels, k = 1)
print(acc)
print(idx)
print(time.time()-time1)


# ### Calculate accuracy for convolutional

# In[23]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def calculate_accuracy_euc_knn_conv(train_images, test_images, train_labels, test_labels, k=4):
    # Reshape the 4D image arrays into 2D arrays
    train_images_reshaped = train_images.reshape(train_images.shape[0], -1)
    test_images_reshaped = test_images.reshape(test_images.shape[0], -1)

    # Initialize the k-NN classifier with Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Train the k-NN classifier on the training data
    knn.fit(train_images_reshaped, train_labels)

    # Predict the labels for the test images
    predictions = knn.predict(test_images_reshaped)

    # Identify correctly classified indices
    correct_indices = [i for i, (pred, true) in enumerate(zip(predictions, test_labels)) if pred == true]

    # Calculate accuracy
    accuracy = (len(correct_indices) / len(test_labels)) * 100

    return accuracy, correct_indices


# In[24]:


# Predict normalized images using the trained autoencoder
train_images_latent_conv = conv_encoder.predict(train_images, batch_size=32, verbose=1)
test_images_latent_conv = conv_encoder.predict(test_images, batch_size=32, verbose=1)

# Verify the shape of the predicted images
print(f"Predicted Images Shape: {train_images_latent_conv.shape}")
print(f"Predicted Images Shape: {test_images_latent_conv.shape}")


# In[25]:


time1 = time.time()
acc_conv,idx_conv = calculate_accuracy_euc_knn_conv(train_images_latent_conv, test_images_latent_conv,train_labels, test_labels, k = 1)
print(acc_conv)
print(idx_conv)
print(f"time taken{time.time()-time1}")


# # Variational implement 

# In[26]:


import tensorflow as tf
from tensorflow.keras import layers, Model

# Define the encoder
class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_logvar = layers.Dense(latent_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        mean = self.dense_mean(x)
        logvar = self.dense_logvar(x)
        return mean, logvar

# Define the decoder
class Decoder(Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense = layers.Dense(28 * 28 * 64, activation="relu")
        self.reshape = layers.Reshape((28, 28, 64))
        self.deconv1 = layers.Conv2DTranspose(64, (3, 3), activation="relu", strides=2, padding="same")
        self.deconv2 = layers.Conv2DTranspose(32, (3, 3), activation="relu", strides=2, padding="same")
        self.deconv3 = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

# Define the VAE
class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        epsilon = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(0.5 * logvar) * epsilon

    def call(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mean, logvar

# Loss function
def compute_loss(x, reconstructed, mean, logvar):
    # Initialize the MSE loss function
    mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    # Compute pixel-wise reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(mse_loss_fn(x, reconstructed), axis=(1, 2)))

    # Compute KL divergence
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    kl_divergence = tf.reduce_mean(kl_divergence)

    return reconstruction_loss + kl_divergence

# Training step
@tf.function
def train_step(vae, x, optimizer):
    with tf.GradientTape() as tape:
        reconstructed, mean, logvar = vae(x)
        loss = compute_loss(x, reconstructed, mean, logvar)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

# Create encoder, decoder, and VAE
latent_dim = 64
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
vae = VAE(encoder, decoder)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

from tensorflow.keras.utils import Progbar

# Training loop with progress bar
epochs = 10
loss_values = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    progbar = Progbar(len(train_dataset))  # Initialize progress bar for the epoch

    for step, train_batch in enumerate(train_dataset):
        train_loss = train_step(vae, train_batch[0], optimizer)
        loss_values.append(train_loss.numpy())  # Store the loss value
        progbar.update(step + 1, [("loss", train_loss.numpy())])  # Update progress bar
        # if step%100 ==0:
        #     print(step)
    print(f"Epoch {epoch + 1} completed with loss: {train_loss.numpy():.4f}")



# In[27]:


# After training, plot the loss values
plt.plot(loss_values)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations')
plt.show()


# In[28]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def calculate_accuracy_euc_knn(train_images, test_images, train_labels, test_labels, k=1):

    # Initialize the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Train the k-NN classifier on the training data
    knn.fit(train_images, train_labels)

    # Predict the labels for the test images
    predictions = knn.predict(test_images)

    # Identify correctly classified indices
    correct_indices = [i for i, (pred, true) in enumerate(zip(predictions, test_labels)) if pred == true]

    # Calculate accuracy
    accuracy = (len(correct_indices) / len(test_labels)) * 100

    return accuracy, correct_indices


# ## accuracy

# In[29]:


# Prediction function to output latent space
def predict_latent_space(dataset, encoder):
    latent_space = []
    for batch in dataset:
        x = batch[0]
        mean, _ = encoder(x)
        latent_space.append(mean.numpy())
    return np.concatenate(latent_space, axis=0)

# Predict latent space representations
train_latent = predict_latent_space(train_dataset, encoder)
test_latent = predict_latent_space(test_dataset, encoder)


# In[30]:


time1 = time.time()
acc_vae,idx_vae = calculate_accuracy_euc_knn(train_latent, test_latent,train_labels, test_labels, k = 1)
print(acc_vae)
print(idx_vae)
print(f"time taken{time.time()-time1}")


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assuming train_labels and test_labels are available
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_latent, train_labels)

# Predict on the test set
predictions = knn.predict(test_latent)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Classification accuracy: {accuracy:.4f}")


# ## misc

# In[ ]:




