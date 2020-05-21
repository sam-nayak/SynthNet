from keras import backend as K
from tensorflow.keras.layers import Layer


class FuzzyLayer(Layer):

    def __init__(self, output_dim, initializer_centers=None, initializer_sigmas=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initializer_centers = initializer_centers
        self.initializer_sigmas = initializer_sigmas
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fuzzy_degree = self.add_weight(name='fuzzy_degree',
                                            shape=(input_shape[-1], self.output_dim),
                                            initializer=self.initializer_centers if self.initializer_centers is not None else 'uniform',
                                            trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(input_shape[-1], self.output_dim),
                                     initializer=self.initializer_sigmas if self.initializer_sigmas is not None else 'ones',
                                     trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, input, **kwargs):
        x = K.repeat_elements(K.expand_dims(input, axis=-1), self.output_dim, -1)

        fuzzy_out = K.exp(-K.sum(K.square((x - self.fuzzy_degree) / (self.sigma ** 2)), axis=-2, keepdims=False))
        return fuzzy_out

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)
