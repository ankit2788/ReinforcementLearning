from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
import os

class ModifiedTensorBoardCallback(TensorBoard):
    # modified Tensorboard class
    # This is written specifically for RL agent, where loggings isnt done at every fit
    # Source: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
    # 

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, model,  **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer     = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()

        hists = []
        modelName = model.name

        self._log_write_dir     = os.path.join(self.log_dir, modelName)
        self._train_dir         = os.path.join(self.log_dir, "train")
        self._train_step        = 0
        self._val_dir           = os.path.join(self.log_dir, "validation")
        self._val_step          = 0
        self._should_write_train_graph = False

        layerCount = 0

        self.model = model
        with self.writer.as_default():
            for layer in model.layers:
                for weight in layer.weights:

                    weightName = weight.name
                    weightName = weightName.split("/")[1].split(":")[0] + "_" + str(layerCount)

                    name = modelName + "_" + weightName
                    tf.summary.histogram(f'weights/{name}', weight, step = self.step)

                    self.writer.flush()

                layerCount += 1

                    



    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # custom writer
    def _write_logs(self, logs, index):

        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step = index)
                self.writer.flush()


    # Custom method for saving own histograms
    # Creates writer, writes custom metrics and closes writer
    def update_stats_histogram(self, model):
        self._write_logs_histogram(model, self.step)


    def  _write_logs_histogram(self, model, index):

        modelName = model.name
        layerCount = 0

        with self.writer.as_default():
            for layer in model.layers:
                for weight in layer.weights:

                    weightName = weight.name
                    weightName = weightName.split("/")[1].split(":")[0] + "_" + str(layerCount)

                    name = modelName + "_" + weightName
                    tf.summary.histogram(f'weights/{name}', weight, step = self.step)

                    self.writer.flush()

                layerCount += 1




