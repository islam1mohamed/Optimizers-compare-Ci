import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Nadam
from tensorflow.keras.callbacks import  LearningRateScheduler
import math
import time

epoch_val = 20

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train= x_train / 255.0
x_test = x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

def shallow_model():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# vars
results = {}
etta_SGD = 0.005
etta_Nadam = 0.00005

with open("results","+a") as f :
    f.writelines(f"etta values SGD , NAG ={etta_SGD}  , etta values RMSProp , Nadam ={etta_Nadam}\n")
        

# Optimizers and learning rate schedulers
optimizers = {
    "SGD": SGD(learning_rate=etta_SGD, momentum=0.9, nesterov=False),
    "NAG": SGD(learning_rate=etta_SGD, momentum=0.9, nesterov=True),
    "RMSProp": RMSprop(learning_rate=etta_Nadam),
    "Nadam": Nadam(learning_rate=etta_Nadam)
}

lr_schedulers = {
    "Exponential Decay": lambda epoch: 0.01 * tf.math.exp(-0.1 * epoch),
    "Step Decay": lambda epoch: 0.01 if epoch < 5 else 0.0005
}


def sgd_warm_restarts(epoch, lr):
    T_0 = epoch_val  # Initial restart period (number of epochs)
    T_mult = 2  # Factor by which T_0 increases after each restart
    max_lr = etta_SGD  # Maximum 
    min_lr = 0.00001  # Minimum 
    cycle = math.floor(1 + epoch / T_0)
    T_cur = epoch - T_0 * (1 - 1 / T_mult) ** (cycle - 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * T_cur / T_0))

# warm restarts
print("Warm Restarts...")

model = shallow_model()
optimizer = SGD(learning_rate= etta_SGD, momentum=0.9)  # Use SGD optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_callback = LearningRateScheduler(sgd_warm_restarts)

start_time = time.time()
history = model.fit(
    x_train, y_train,
    epochs=epoch_val,
    validation_data=(x_test, y_test),
    callbacks=[lr_callback],  # attach warm restarts scheduler
    verbose=1
)
end_time = time.time()

# Save results
results["SGD with Warm Restarts"] = {
    "Training Time": end_time - start_time,
    "Validation Accuracy": max(history.history['val_accuracy'])
}

# Train with each optimizer and scheduler
for name, optimizer in optimizers.items():
    print(f"Training with {name} optimizer...")
    model = shallow_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epoch_val,  # high more train more time
        validation_data=(x_test, y_test),
        verbose = 1
    )
    
    end_time = time.time()
    
    results[name] = {
        "Training Time": end_time - start_time,
        "Validation Accuracy": max(history.history['val_accuracy'])
    }


for name, scheduler in lr_schedulers.items():
    print(f"Training with {name} scheduler...")
    model = shallow_model()
    model.compile(optimizer=SGD(learning_rate=etta_SGD), loss='categorical_crossentropy', metrics=['accuracy'])
    
    lr_callback = LearningRateScheduler(scheduler)
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epoch_val, 
        validation_data=(x_test, y_test),
        verbose = 1,  # Show training progress
        callbacks=[lr_callback]  # Pass the scheduler here

    )
    end_time = time.time()
    results[name] = {
        "Training Time": end_time - start_time,
        "Validation Accuracy": max(history.history['val_accuracy'])
    }

# Print Results
print("\nFinal Results:")
for key, value in results.items():
    print(f"{key}: {value}")
    with open("results","+a") as f :
        f.writelines(f"{key}: {value}\n")
with open("results","+a") as f :
    f.writelines("--------------------------------\n")
