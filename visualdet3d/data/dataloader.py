import tensorflow as tf


class DataLoader:
    def __init__(self, dataset):
        pass
    
    def _gen_dataset(self):
        self.data = tf.data.Dataset.from_generator()