from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
from tensorflow.keras import backend as K
 
class ModifiedTensorBoardCallback(TensorBoard):
    # modified Tensorboard class
    # This is written specifically for RL agent, where loggings isnt done at every fit
    # Source: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
    # 

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, model,  **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer     = tf.summary.FileWriter(self.log_dir)

        hists = []
        modelName = model.name
        for layer in model.layers:
            for weight in layer.weights:

                name = modelName + weight.name.replace(":","_")
                histogram = tf.summary.histogram(f'weights/{name}', weight)
                hists.append(histogram)

                    
        merged = tf.summary.merge(hists)
        self.histogram = merged


        #hist = K.get_session().run(merged)


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
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()


    # Custom method for saving own histograms
    # Creates writer, writes custom metrics and closes writer
    def update_stats_histogram(self, **stats):
        self._write_logs_histogram(self.step)


    def  _write_logs_histogram(self, index):

        hist = K.get_session().run(self.histogram)

        self.writer.add_summary(hist, index)

        self.writer.flush()






class GradCallBack(TensorBoard):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.step       = 1
        self.writer     = tf.summary.FileWriter(self.log_dir)
        self.model      = model



    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        # logs: Dict.
        # contains gradients of loss function with respect to the model 
        pass
        #self.update_grads(**logs)


    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def update_grads(self, gradStats):
        self._write_logs(gradStats, self.step)

    # custom writer
    def _write_logs(self, logs, index):
        # logs: Dict.
        # contains gradients of loss function with respect to the model 

        hists = []
        for gradName in logs.keys():

            curr_grad = logs[gradName][0]
            if curr_grad is not None:
                
                _thisHist = tf.summary.histogram(f'grad_histogram_/{gradName}', curr_grad)
                hists.append(_thisHist)

        if len(hists) > 0:
            merged      = tf.summary.merge(hists)
            histogram   = merged
            

            runHist     = K.get_session().run(histogram)

            self.writer.add_summary(runHist, index)
            self.writer.flush()
