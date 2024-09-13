import numpy as np
import tensorflow as tf

def split_dataset(f, response, label, test_ratio=0.2):
    """
    Split the data into training and testing sets.

    Args:
        f (array-like, shape [n_samples, n_features]): Predicted neural response.
        response (array-like, shape [n_samples, n_features]): True neural response.
        label (array-like, shape [n_samples, n_labels]): Input data (e.g., positions).
        test_ratio (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Tuple containing training and testing data. Each element of the tuple 
               is a tuple (f, response, label) for train and test splits respectively.
    """
    total_size = len(f)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    
    test_size = int(total_size * test_ratio)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    f_train = f[train_indices]
    response_train = response[train_indices]
    label_train = label[train_indices]
    f_test = f[test_indices]
    response_test = response[test_indices]
    label_test = label[test_indices]
    
    return (f_train, response_train, label_train), (f_test, response_test, label_test)

def create_dataset(f, response, label, batch_size):
    """
    Create a TensorFlow dataset from the given data, with data types converted to tf.float32.

    Args:
        f (array-like, shape [n_samples, n_features]): Predicted neural response.
        response (array-like, shape [n_samples, n_features]): True neural response.
        label (array-like, shape [n_samples, n_labels]): Input data (e.g., positions).
        batch_size (int): Batch size for the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing batches of (f, response, label), all as tf.float32.
    """
    f = tf.cast(f, tf.float32)
    response = tf.cast(response, tf.float32)
    label = tf.cast(label, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((f, response, label))
    dataset = dataset.shuffle(buffer_size=len(f)).batch(batch_size)
    return dataset

def shuffle_fire_rate(fire_rate, x, y, speed, x_bin, y_bin, speed_bin):
    '''
    inputs:
        fire_rate: (n_samples, n_units)
        x: (n_samples, 1)
        y: (n_samples, 1)
        speed: (n_samples, 1)
        x_bin: (n_bins + 1,)
        y_bin: (n_bins + 1,)
        speed_bin: (n_bins + 1,)
    outputs:
        shuffled_fire_rate: (n_samples, n_units)
    # Example usage
    # Please replace the following with your actual data
    n_samples = 100  # number of samples
    n_units = 5      # number of units
    n_bins = 4       # number of bins

    # Dummy data
    fire_rate = np.random.rand(n_samples, n_units)
    x = np.random.rand(n_samples, 1)
    y = np.random.rand(n_samples, 1)
    speed = np.random.rand(n_samples, 1)
    x_bin = np.linspace(0, 1, n_bins + 1)
    y_bin = np.linspace(0, 1, n_bins + 1)
    speed_bin = np.linspace(0, 1, n_bins + 1)

    # Shuffle the fire_rate
    shuffled_fire_rate = shuffle_fire_rate(fire_rate, x, y, speed, x_bin, y_bin, speed_bin)
    shuffled_fire_rate.shape  # Check the shape of the shuffled fire_rate
    '''
    # Assuming fire_rate, x, y, speed are numpy arrays and x_bin, y_bin, speed_bin are bin edges.
    n_samples, n_units = fire_rate.shape

    # Binning the samples
    x_indices = np.digitize(x, x_bin) - 1
    y_indices = np.digitize(y, y_bin) - 1
    speed_indices = np.digitize(speed, speed_bin) - 1

    # Creating a shuffled fire_rate array with the same shape
    shuffled_fire_rate = np.zeros_like(fire_rate)

    # Iterate through each bin and shuffle the columns
    for i in range(len(x_bin)):
        for j in range(len(y_bin)):
            for k in range(len(speed_bin)):
                # Selecting the samples belonging to the current bin
                indices = np.where((x_indices == i) & (y_indices == j) & (speed_indices == k))[0]
                if len(indices) > 0:
                    bin_fire_rate = fire_rate[indices, :]
                    
                    # Shuffling each column
                    for col in range(n_units):
                        np.random.shuffle(bin_fire_rate[:, col])
                    
                    # Assigning the shuffled values back
                    shuffled_fire_rate[indices, :] = bin_fire_rate

    return shuffled_fire_rate
