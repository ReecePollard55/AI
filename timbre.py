import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow_core
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Input, Dropout
from tensorflow_core.python.keras.layers.preprocessing.normalization import Normalization
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    target = 'timbre'
    output_path = 'spotify_M'

    X_train, y_train = get_xy('spotify_train.npz', target)
    X_valid, y_valid = get_xy('spotify_valid.npz', target)

    keras.backend.clear_session()
    # setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # create model
    model = Sequential()
    #model.add(Input(129))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.fit(X_train, y_train, verbose=1, validation_data=[X_valid, y_valid],
              callbacks=[model_checkpoint, early_stopping], epochs=50, batch_size=128)
    model.summary()
    model = get_best_model(output_path)

    visualize(model, X=X_train, y=y_train, name='Training', output_path=output_path)
    visualize(model, X=X_valid, y=y_valid, name='Validation', output_path=output_path)



# ignore
def get_xy(npz_file, endpoint):
    """
    Gets the data matrix, X, and target, y for the desired 'endpoint'.

    :param npz_file: The path to the npz file
    :param endpoint: 'pitches', 'timbre', or 'loudness'
    :return:
    """
    with np.load(npz_file) as data:
        x = data['x']
        key = f'y_{endpoint}'
        if key not in data:
            raise ValueError(f'Unknown endpoint {endpoint}')
        y = data[key]
        if y.ndim == 1:
            y = data[key].reshape((-1, 1))
    return x, y

# ignore
def setup_model_checkpoints(output_path, save_freq):
    """
    Setup model checkpoints using the save path and frequency.

    :param output_path: The directory to store the checkpoints in
    :param save_freq: The frequency with which to save them "epoch" means each epoch
                      See ModelCheckpoint documentation.
    :return: a ModelCheckpoint
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_path, 'model.{epoch:05d}_{val_loss:f}.h5'),
        save_weights_only=False,
        save_freq=save_freq,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    return model_checkpoint


# ignore
def visualize(model, X, y, name='', output_path=''):
    """
    Create a joint distribution plot that shows relationship between
    model estimates and true values.

    :param model: A trained model
    :param X: The data matrix, X
    :param y: The target vector or matrix, y
    :param name: The name for the figure
    :param output_path: The output directory to save the PNG
    :return: None
    """
    png_file = os.path.join(output_path, f'visualize_{name}.png')
    q = model.predict(X)
    loss = model.evaluate(X, y)
    y = y.reshape((-1,))
    q = q.reshape((-1,))
    jg = sns.jointplot(x=y, y=q, kind='hist')
    jg.fig.suptitle(f'{name} (loss = {loss:.6f})')
    jg.set_axis_labels(xlabel='Actual', ylabel='Model')
    max_value = max(q.max(), y.max())
    min_value = min(q.min(), y.min())
    jg.ax_joint.plot([min_value, max_value], [min_value, max_value], color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(png_file)
    plt.close(jg.fig)


# ignore
def get_best_model(output_path):
    """
    Parses the output_path to find the best model. Relies on the ModelCheckpoint
    saving a file name with the validation loss in it. If a model was saved with
    a Normalization layer, it's provided as a custom object.

    :param output_path: The directory to scan for H5 files
    :return: The best model compiled.
    """
    min_loss = float('inf')
    best_model_file = None
    for file_name in os.listdir(output_path):
        if file_name.endswith('.h5'):
            val_loss = float('.'.join(file_name.split('_')[1].split('.')[:-1]))
            if val_loss < min_loss:
                best_model_file = file_name
                min_loss = val_loss
    print(f'loading best model: {best_model_file}')
    model = keras.models.load_model(os.path.join(output_path, best_model_file), compile=True,
                                    custom_objects={'Normalization': Normalization})
    return model

# ignore
def loudness_example():
    """
    An example applying linear regression to the loudness problem.

    :return: None
    """
    target = 'loudness'
    output_path = 'spotify_loudness_linear'

    X_train, y_train = get_xy('C:/Users/Reece/PycharmProjects/pythonProject32/spotify_train.npz', target)
    X_valid, y_valid = get_xy('C:/Users/Reece/PycharmProjects/pythonProject32/spotify_train.npz', target)

    keras.backend.clear_session()
    # setup callbacks
    model_checkpoint = setup_model_checkpoints(output_path, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # create linear model
    model = Sequential()
    model.add(Input(129))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    model.fit(X_train, y_train, verbose=1, validation_data=[X_valid, y_valid],
              callbacks=[model_checkpoint, early_stopping])

    model = get_best_model(output_path)

    visualize(model, X=X_train, y=y_train, name='Training', output_path=output_path)
    visualize(model, X=X_valid, y=y_valid, name='Validation', output_path=output_path)

train_model()