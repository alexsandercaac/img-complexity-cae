"""
    Module with auxiliary function for Keras models.
    such as custom Keras callbacks to log information during training, as well
    as control of the learning rate schedule.

"""
from numpy.core.numeric import Inf
from tqdm import tqdm
from typing import Union

import tensorflow as tf


def randomize_model_weigths(model: tf.keras.Model) -> None:
    """Randomize the weights of a Keras model.

    Args:
        model (tf.keras.Model): model to be randomized.
    Returns:
        None
    """
    model.set_weights(
        [tf.random.normal(shape=weight.shape) for weight in model.weights]
    )


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """

    Learning rate scheduler with additional functionalities.

    The learning rate is controlled by two constants: alpha and beta. These
    constants will decay the learning rate when there is improvement in the
    validation metric and where there is not, respectively. Intuitively,
    you would expect alpha to be greater than beta, since alpha is employed
    when the optimizer is "on the right path" and val error is decreasing.

    If no improvement is observed in validation, the training can either be
    stopped early or the model which yielded the best metric is "revived"
    and training is resumed from that point, but with a smaller learning
    rate, aiming at local exploration of the loss curve.

  Args:
        alpha (float): scaling to be applied to the learning rate when there
            was improvement in the metric between current and last epoch.
        beta (float): scaling to be applied to the learning rate when there was
            NO improvement in the metric.
        metric (string): metric evaluated for determining the scaling of the
            learning rate.
        min_lr (float): lower threshold for the learning rate.
        return_best_weights (bool): if True, weights associated with epoch of
            best main metric are set to the model.
        early_stopping (bool): if True, stop training if "patience" epochs
            without improvemnt are reached. IGNORED if
            revive_best is True.
        revive_best (bool): if True, reset model weights to the best epoch and
            proceed training with smaller learning rate.
            Used for exploiting, after exploring is done.
        maximize (bool): if true, main metric is to be maximized.
        patience (int): epochs to wait before revive or early stopping; ignored
            if both revive_best and early_stopping are set to False
        secondary_metric (string): secondary metric to keep track of.
        bin_checkpoint (bool): if True, save model weights in binary format.
        bin_path (string): path to save binary checkpoint.
        verbose (int): if < 1, supress most messages; if 1, present some
            messages; if >1, present all messages.
        initial_metric (float): value to initialize the metric. Useful when
            using a pretrained model. If None, the metric is initialized
            with the +-Inf value, depending on the maximize flag.
  """

    def __init__(self, alpha: float = 0.98, beta: float = 0.9,
                 metric: str = 'val_loss', min_lr: float = 1e-5,
                 return_best_weights: bool = True,
                 early_stopping: bool = False, maximize: bool = False,
                 revive_best: bool = False, patience: Union[int, float] = Inf,
                 secondary_metric: str = '', bin_checkpoint: bool = False,
                 bin_path: str = '', verbose: int = 0,
                 initial_metric: float = None
                 ):
        super(CustomLearningRateScheduler, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.metric = metric
        self.min_lr = min_lr
        self.maximize = maximize
        self.verbose = verbose
        self.revive_best = revive_best
        self.early_stopping = early_stopping
        self.patience = patience
        self.return_best_weights = return_best_weights
        self.best_metric = Inf
        self.best_secondary = Inf
        self.best_weights = None
        self.secondary_metric = secondary_metric
        self.lr_at_min = False
        self.bin_checkpoint = bin_checkpoint
        if bin_checkpoint:
            self.bin_path = bin_path
        self.summary_history = {'lr': [], 'logs': []}
        self.__initial_metric = initial_metric

    def on_train_begin(self, logs=None):
        self.__no_improv_epochs = 0
        self.stopped_epoch = 0
        self.__metric_value = self.__initial_metric
        # If an initial metric is provided, the model is assumed to be
        # pretrained and the initial metric is used to initialize the
        # best metric.
        if self.__initial_metric:
            lr = float(tf.keras.backend.get_value(
                self.model.optimizer.learning_rate))
            self.summary_history['lr'].append(lr)
            self.best_metric = self.__metric_value
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            tqdm.write(f"Epoch {(self.stopped_epoch + 1)}: early stopping")
        if self.revive_best:
            if self.verbose > 0:
                tqdm.write("Restoring model weights from the end of" +
                           " the best epoch")
            self.model.set_weights(self.best_weights)
            if self.__initial_metric:
                first_log = self.summary_history['logs'][0]
                first_log = {
                    key: None if key != self.metric else self.__initial_metric
                    for key in first_log.keys()}

                self.summary_history['logs'] = (
                    [first_log] + self.summary_history['logs'])

    def on_epoch_end(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute')
        if self.metric not in logs:
            raise ValueError('Metric not found in training logs')
        if self.secondary_metric:
            if self.secondary_metric not in logs:
                raise ValueError('Secondary metric not found' +
                                 'in training logs')
        # Get the current learning rate from optimizer.
        lr = float(tf.keras.backend.get_value(
                   self.model.optimizer.learning_rate))
        self.summary_history['lr'].append(lr)
        self.summary_history['logs'].append(logs)

        if epoch == 0:
            # First epoch, no previous metric to compare
            if self.__metric_value:
                # Compare current metric with last logged metric
                metric_diff = logs[self.metric] - self.__metric_value
                # Last logged metric is now current metric
                self.__metric_value = logs[self.metric]

                if self.maximize:
                    metric_diff = -1*metric_diff
                    self.__metric_value = -1*self.__metric_value

                # If there was improvement
                if metric_diff < 0:
                    scheduled_lr = lr*self.alpha    # Small decay
                    self.best_metric = self.__metric_value
                    self.best_weights = self.model.get_weights()
                else:
                    scheduled_lr = lr*self.beta
                set_lr(scheduled_lr, epoch, self.model)

            else:
                self.__metric_value = logs[self.metric]
                if self.bin_checkpoint:
                    self.model.save(self.bin_path)
                if self.verbose > 0:
                    tqdm.write(f"Epoch: {epoch+1} - New best model!! :)")
                self.best_metric = self.__metric_value
                self.best_weights = self.model.get_weights()
                if self.secondary_metric:
                    self.best_secondary = logs[self.secondary_metric]
                scheduled_lr = self.alpha*lr
                set_lr(scheduled_lr, epoch, self.model)

        else:
            # Compare current metric with last logged metric
            metric_diff = logs[self.metric] - self.__metric_value
            # Last logged metric is now current metric
            self.__metric_value = logs[self.metric]

            if self.maximize:
                metric_diff = -1*metric_diff
                self.__metric_value = -1*self.__metric_value

            # If there was improvement
            if metric_diff < 0:
                scheduled_lr = lr*self.alpha    # Small decay
            else:
                scheduled_lr = lr*self.beta

            # If new metric is the best so far
            if self.__metric_value < self.best_metric:
                self.__no_improv_epochs = 0
                if self.bin_checkpoint:
                    self.model.save(self.bin_path)
                if self.verbose > 0:
                    tqdm.write(f"\nEpoch: {epoch+1} - New best model!! :)")
                self.best_metric = self.__metric_value
                self.best_weights = self.model.get_weights()
                if self.secondary_metric:
                    self.best_secondary = logs[self.secondary_metric]
            else:
                self.__no_improv_epochs += 1
                if self.verbose > 1:
                    tqdm.write(f"{self.__no_improv_epochs} epochs" +
                               " without improvement")
                # Check if patience has run out.
                if self.__no_improv_epochs > self.patience:
                    if self.revive_best:
                        if self.verbose > 0:
                            tqdm.write("Reviving best model")
                        self.model.set_weights(self.best_weights)
                        self.__no_improv_epochs = 0
                    elif self.early_stopping:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        return

            if scheduled_lr > self.min_lr:
                set_lr(scheduled_lr, epoch, self.model, self.verbose)

            else:
                if self.lr_at_min is False:
                    set_lr(self.min_lr, epoch, self.model, self.verbose)
                    self.lr_at_min = True
                    if self.verbose > 0:
                        tqdm.write("Learning rate at minimum")


def set_lr(lr, epoch, model, verbose=1):
    tf.keras.backend.set_value(model.optimizer.lr, lr)
    if verbose > 1:
        tqdm.write("\nEpoch %d: Learning rate is %f." % (epoch+2, lr))


class LoggingCallback(tf.keras.callbacks.Callback):
    """
    ### Summary


  """

    def __init__(self, metric: str = 'val_loss', secondary_metric: str = '',
                 verbose: int = 0
                 ):
        super(LoggingCallback, self).__init__()
        self.metric = metric
        self.verbose = verbose
        self.secondary_metric = secondary_metric
        self.summary_history = {'lr': [], 'logs': []}

    def on_epoch_end(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute')
        if self.metric not in logs:
            raise ValueError('Metric not found in training logs')
        if self.secondary_metric:
            if self.secondary_metric not in logs:
                raise ValueError('Secondary metric not found' +
                                 'in training logs')
        # Get the current learning rate from optimizer.
        lr = float(tf.keras.backend.get_value(
                   self.model.optimizer.learning_rate))

        self.summary_history['lr'].append(lr)
        self.summary_history['logs'].append(logs)
