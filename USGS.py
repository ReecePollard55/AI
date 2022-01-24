import numpy as np
import skimage
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage import data, io, filters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import SGD


def train():
    image = 'C:/Users/Reece/PycharmProjects/pythonProject32/USGS_NED_1_n38w083_IMG.tif'
    x_train, y_train = get_xy(image, 'crop')
    # y_train = y_train.astype('float32') / y_train.max()
    # x_train = x_train.astype('float32') / x_train.max()

    y_max = y_train.max()

    model = Sequential()
    model.add(Dense(128, input_dim=2, activation='relu'))
    model.add(Dense(16, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Multiply output because last layer is sigmoid
    model.add(Lambda(lambda y: y * y_max))

    # train model
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(x_train, y_train, epochs=1500, batch_size=256)

    model.summary()
    print('----------------------------------------------------------\nSaving model..')
    model.save('n38w083_crop.h5')
    print('----------------------------------------------------------\n')
    compare_images(model, x_train, y_train)




def get_xy(tiff_file, type_):
    """
    Read, trim, and crop or down-sample TIFF file and reshape into two-dimensional
    data matrix, X, and target vector, y.
    :param tiff_file: Image file name (string)
    :param type_: 'crop' to crop the northwest corner of the image,
                  'ds' to down-sample by skipping every 10 pixels
    :return: data matrix 'X' (129600-by-2), target vector 'y' (129600-by-1)
    """
    # Read the image
    image = skimage.io.imread(tiff_file)

    # Trim the first and last 6 rows and columns
    image = image[6:-6, 6:-6]

    if type_ == 'crop':
        print("Cropping image")
        # crop to the first 360 rows and columns
        image = image[:360, :360]
    elif type_ == 'ds':
        # down-sample by keeping every 10th row and column
        image = image[::10, ::10]
    else:
        # if a bad type is used
        raise ValueError(f'Unknown test type: {type_}')

    # get the row index 'i' and column index 'j' for every pixel
    # (why start at -180 and not zero?)
    i, j = np.meshgrid(range(-180, 180), range(-180, 180), indexing='ij')

    # reshape row/column indexes into n-by-2 matrix
    x = np.concatenate((i.reshape((-1, 1)), j.reshape((-1, 1))), axis=1)

    # reshape elevations into n-by-1 vector
    y = image.reshape((-1, 1))

    return x, y


def error_bits(error):
    """
    Return a lower bound on the number of bits to encode the errors based on Shannon's source coding theorem.
    :param error: Vector or list of errors (error = estimate - actual)
    :return: The lower bound number of bits to encode the errors
    """
    # round and cast to an integer, reshape as a vector
    error = np.round(error).astype(int).reshape((-1,))

    # shift so that the minimum value is zero
    error = error - error.min()

    # count how many occurrences of each discrete value
    p = np.bincount(error)

    # ignore zero counts
    p = p[p > 0]

    # convert counts into discrete probability distribution
    p = p / p.sum()

    # compute entropy (bits per codeword)
    entropy = -(p * np.log2(p)).sum()

    # minimum bits to encode all errors
    bits = int(np.ceil(entropy * len(error)))

    return bits


def compare_images(model, x, y):
    """
    Use the model to estimate elevation image and show it compared
    to the original.
    :param model: A keras model with 2D input and 1 output
    :param x: The (i, j) coordinates
    :param y: The target vector
    :return: None
    """
    # The model estimates (same shape as 'y')
    y_hat = model.predict(x)
    error = y_hat - y
    mse = np.mean(error ** 2)
    err_bits = error_bits(error)
    mod_bits = 32 * model.count_params()
    orig_bits = error_bits(y)
    # print some comparisons
    # print_error(y, 0, 'Zero')
    # print_error(y.mean() - y, 1, 'Constant')
    # print_error(LinearRegression().fit(x, y).predict(x) - y, 3, 'Linear')
    # print_error(y_hat - y, model.count_params(), 'Model')
    improvement = 1 - (mod_bits + err_bits) / orig_bits
    # The minimum and maximum elevation over original and estimated image
    vmin = min(y.min(), y_hat.min())
    vmax = max(y.max(), y_hat.max())
    # The (i, j) extents (-180 to 179)
    extent = (-180.5, 179.5, 179.5, -180.5)
    # close all figure windows
    plt.close('all')
    # create a new figure window with room for 2 adjacent images and a color bar
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
    # render and label the model estimate
    im0 = ax[0].imshow(y_hat.reshape((360, 360)), vmin=vmin, vmax=vmax, extent=extent)
    ax[0].set_title(f'Model Estimate MSE = {mse:.1f}\n'
                    f'entropy = {err_bits / len(y):.4f}, model_bits/px = {mod_bits / len(y):.4f}\n'
                    f'total bits/px = {(err_bits + mod_bits) / len(y):.4f}, '
                    f'improvement = {improvement:.2%}')
    # render and label the original
    im1 = ax[1].imshow(y.reshape((360, 360)), vmin=vmin, vmax=vmax, extent=extent)
    ax[1].set_title('Original')
    # add a color bar in a new set of axes
    fig.subplots_adjust(right=0.8)
    ax2 = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=ax2)
    # make the figure visible
    plt.show()


train()
