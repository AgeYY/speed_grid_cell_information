import tensorflow as tf
import gpflow
from tqdm import tqdm
import os
from datetime import datetime
from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

def run_adam(model, data, iterations, learning_rate=0.01, minibatch_size=25, natgrads=False, pb=True):
    """
    Utility function running the Adam optimizer.

    :param model: GPflow model
    :param data: tuple (X,Y)
    :param interations (int) number of iterations
    :param learning_rate (float)
    :param minibatch_size (int)
    :param natgrads (bool) Optimize variational parameters with natural gradient if True
    :param plot (bool) Plot loss convergence if true.
    """
    logf = []
    if natgrads:
        variational_params = [(model.q_mu, model.q_sqrt)]
        for param in variational_params:
            gpflow.set_trainable(param, False)
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.001)

    ## mini batches
    N, D = data[0].shape
    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N) # minibatch data
    train_iter = iter(train_dataset.batch(minibatch_size))

    ## one data batch
    # train_iter = tuple(map(tf.convert_to_tensor, data))

    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # tensorboard logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.abspath((os.path.join(os.getcwd(),
                                            '..',
                                            'logs',
                                            timestamp)))
    log_dir = str(log_dir)

    model_task = ModelToTensorBoard(log_dir, model)
    elbo_task = ScalarToTensorBoard(log_dir, lambda: -training_loss().numpy(), "ELBO")

    short_period, long_period = 10, 100
    tasks = MonitorTaskGroup([elbo_task, model_task], period=short_period)
    monitor = Monitor(tasks)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        if natgrads:
            natgrad_opt.minimize(training_loss, var_list=variational_params)

    for step in tqdm(range(iterations), disable=not pb):
        optimization_step()
        monitor(step)

        if step % long_period == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)

    return logf