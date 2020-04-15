from keras import backend as K


def sparse_masked_mlm_loss(y_true, y_pred):
    mask = K.cast(K.any(y_true, axis=-1), "float32")
    cce = K.sparse_categorical_crossentropy(y_true, y_pred)
    masked_cce = mask * cce
    return K.sum(masked_cce) / (K.sum(mask) + K.epsilon())


def masked_loss(y_true, y_pred):
    y_mask = K.cast(K.any(y_true, axis=-1), "float32")
    loss = K.switch(y_mask,
                    K.sparse_categorical_crossentropy(y_true,
                                                      y_pred),
                    K.zeros_like(y_mask, dtype=K.floatx()))
    return K.sum(loss) / (K.cast(K.sum(y_mask), dtype='float32') + K.epsilon())
