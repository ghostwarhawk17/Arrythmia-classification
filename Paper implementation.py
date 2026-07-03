import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD

def build_scenario_2_model():
    """
    Builds the 'Scenario 2' automatic optimization CNN model
    from the paper: 'Enhancing arrhythmia prediction through an adaptive 
    deep reinforcement learning framework for ECG signal analysis'.
    """
    model = Sequential([
        Input(shape=(200, 1), name='conv1d_2_input'),
        Conv1D(filters=32, kernel_size=3, activation='softplus', name='conv1d_2'),
        Conv1D(filters=32, kernel_size=3, activation='softplus', name='conv1d_3'),
        MaxPooling1D(pool_size=2, strides=2, name='max_pooling1d_1'),
        Dropout(rate=0.2, name='dropout_1'),
        Flatten(name='flatten_1'),
        Dense(units=512, activation='softplus', name='dense_3'),
        Dense(units=1024, activation='softplus', name='dense_4'),
        Dense(units=6, activation='softmax', name='dense_5')
    ])
    
    # Hyperparameters from Table 3 (Scenario 2)
    learning_rate = 0.002
    # Assuming SGD with momentum based on Section 4.1 'm \in {0.5 - 0.9}'
    momentum = 0.9 
    
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("Building Arrhythmia Prediction Model (Scenario 2)...")
    model = build_scenario_2_model()
    
    # Paper Hyperparameters for training:
    batch_size = 64
    epochs = 15
    
    print(f"Training Hyperparameters:")
    print(f"- Batch Size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Dropout: 0.2")
    print(f"- Learning Rate: 0.002")
    print(f"- Activation: SoftPlus\n")
    
    model.summary()
