from abc import ABC, abstractmethod
import numpy as np

class Transformer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x):
        pass


class MinTransformer(Transformer):
    def __init__(self, min=0):
        self.min=min

    def transform(self, x):
        return np.maximum(self.min, x)

    def inverse_transform(self, x):
        return np.maximum(self.min, x)

class LogTransformer(Transformer):
    def __init__(self, min=0):
        self.min=min

    def transform(self, x):
        return np.log1p(np.maximum(self.min, x))

    def inverse_transform(self, x):
        return np.maximum(self.min, np.exp(x)-1)

class MinMaxTransformer(Transformer):
    def __init__(self, min_scale = 0.0, max_scale = 1.0, range_scale = None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        if range_scale == None:
            self.range_scale = max_scale - min_scale
        else:
            self.range_scale = range_scale

    def transform(self, x):
        self.minimum = np.min(x)
        self.maximum = np.max(x)
        self.range = self.maximum - self.minimum
        return ((x-self.minimum)*self.range_scale/self.range)+self.min_scale

    def inverse_transform(self, x):
        return ((x-self.min_scale)*self.range/self.range_scale)+self.minimum

class ZTransformer(Transformer):
    def __init__(self):
        pass

    def transform(self, x):
        self.mean = np.mean(x)
        self.sd = np.std(x)
        return (x-self.mean)/self.sd

    def inverse_transform(self, x):
        return x*self.sd + self.mean


class CustomTransformer(Transformer):
    def __init__(self, transform_fn, inverse_transform_fn):
        self.transform_fn = transform_fn
        self.inverse_transform_fn = inverse_transform_fn

    def transform(self, x):
        return x.apply(lambda i: self.transform_fn(i))

    def inverse_transform(self, x):
        return x.apply(lambda i: self.inverse_transform_fn(i))
