import keras.backend as K
import tensorflow as tf
from keras.legacy import interfaces
from keras.optimizers import Optimizer


class LAMB(Optimizer):

    def __init__(self, lr=0.00176, beta_1=0.9, beta_2=0.999, epsilon=1e-6,
                 weight_decay=0., accum_iters=1, **kwargs):
        super(LAMB, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        self.lr = K.variable(lr, name='lr')
        self.epsilon = epsilon if epsilon is not None else 1e-6
        self.weight_decay = weight_decay
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

        self.updates = [K.update_add(self.iterations, 1)]

        completed_updates = K.cast(tf.floordiv(self.iterations,
                                               self.accum_iters),
                                   K.floatx())
        t = completed_updates + 1

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v, tg in zip(params, grads, ms, vs, gs):
            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = m * self.beta_1 + (1. - self.beta_1) * avg_grad
            v_t = v * self.beta_2 + (1. - self.beta_2) * K.square(avg_grad)

            m_hat = m_t / (1. - K.pow(self.beta_1, t))
            v_hat = v_t / (1. - K.pow(self.beta_2, t))

            u = m_hat / (K.sqrt(v_hat) + self.epsilon) + self.weight_decay * p

            r = K.sqrt(K.sum(K.square(p))) / K.sqrt(K.sum(K.square(u)))

            p_t = p - self.lr * r * u

            self.updates.append(K.update(m, (1 - update_switch) * m +
                                         update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v +
                                         update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            self.updates.append(K.update(p, (1 - update_switch) * p +
                                         update_switch * p_t))

            return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'weight_decay': self.weight_decay}
        base_config = super(LAMB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""
Thanks to:
https://github.com/keras-team/keras/issues/3556 (alexeydevederkin)
"""


class ADAM(Optimizer):

    def __init__(self, lr=0.0001, beta_1=0.9, beta_2=0.98,
                 epsilon=1e-6, decay=0., amsgrad=False,
                 accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(ADAM, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(tf.floordiv(self.iterations,
                                               self.accum_iters),
                                   K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(
                avg_grad)  # X

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) *
                                             vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) *
                                         m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) *
                                         v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) *
                                         sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) *
                                         p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(ADAM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
