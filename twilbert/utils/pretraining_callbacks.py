from keras.callbacks import Callback
from datetime import datetime


class TimeCheckpoint(Callback):

    def __init__(self, hours_step, path):
        super().__init__()
        self.hours_step = hours_step
        self.prev_time = datetime.utcnow()
        self.act_time = datetime.utcnow()
        self.path = path
        self.hours = 0

    def on_batch_end(self, batch, logs=None):
        self.act_time = datetime.utcnow()
        diff_hours = (self.act_time-self.prev_time).seconds / 3600
        if diff_hours >= 1:
            self.hours += 1
            self.prev_time = self.act_time
            self.act_time = datetime.utcnow()
            self.model.save(self.path + "/chekpoint_%d_hours.hdf5" %
                            self.hours)
