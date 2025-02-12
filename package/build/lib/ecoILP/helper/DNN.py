#!pip install keras-tuner

import tensorflow as tf #print(tf.__version__) #2.7.0 win, 2.9.1 macos
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch, BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.model_selection import GroupShuffleSplit
# from keras import Sequential
# from keras.layers import Dense, Input, Dropout

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def balanced_accuracy(y_true, y_pred, mask=None):
    """
    Compute the balanced accuracy for binary classification.
    It is defined as the average of recall obtained on each class.
    """

    # Apply the mask
    if mask is not None: # TO DO
        ignore = y_true.index.isin(mask)
        y_true = y_true[~ignore]
        y_pred = y_pred[~ignore]

    y_true_rounded = K.round(y_true)
    y_pred_rounded = K.round(y_pred)

    true_positives = K.sum(y_true_rounded * y_pred_rounded)
    true_negatives = K.sum((1 - y_true_rounded) * (1 - y_pred_rounded))
    false_positives = K.sum((1 - y_true_rounded) * y_pred_rounded)
    false_negatives = K.sum(y_true_rounded * (1 - y_pred_rounded))

    recall_0 = true_negatives / (true_negatives + false_positives + K.epsilon())
    recall_1 = true_positives / (true_positives + false_negatives + K.epsilon())
    
    balanced_acc = (recall_0 + recall_1) / 2.0
    
    return balanced_acc

# from keras.metrics import Metric
# class BalancedAccuracy(Metric):

#     def __init__(self, name="balanced_accuracy", mask=None, **kwargs):
#         super(BalancedAccuracy, self).__init__(name=name, **kwargs)
#         self.mask = mask
#         self.balanced_accuracy = self.add_weight(name="bac", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Apply the mask if available
#         if self.mask is not None:
#             mask = K.constant(self.mask, dtype=bool)
#             y_true = K.boolean_mask(y_true, mask)
#             y_pred = K.boolean_mask(y_pred, mask)

#         y_true_rounded = K.round(y_true)
#         y_pred_rounded = K.round(y_pred)

#         true_positives = K.sum(y_true_rounded * y_pred_rounded)
#         true_negatives = K.sum((1 - y_true_rounded) * (1 - y_pred_rounded))
#         false_positives = K.sum((1 - y_true_rounded) * y_pred_rounded)
#         false_negatives = K.sum(y_true_rounded * (1 - y_pred_rounded))

#         recall_0 = true_negatives / (true_negatives + false_positives + K.epsilon())
#         recall_1 = true_positives / (true_positives + false_negatives + K.epsilon())
        
#         balanced_acc = (recall_0 + recall_1) / 2.0
#         self.balanced_accuracy.assign(balanced_acc)
        
#     def result(self):
#         return self.balanced_accuracy

#     def reset_states(self):
#         self.balanced_accuracy.assign(0.0)

def create_model_cv(hp, input_dim=-1):
    if input_dim == -1: # TO DO
        input_dim = 104 # or error
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units_1', min_value=8, max_value=128, step=8),
                           activation=hp.Choice('activation_1', values=['relu', 'tanh', 'sigmoid']),
                           input_dim=input_dim))
    for i in range(hp.Int('hidden_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int(f'units_{i+2}', min_value=8, max_value=128, step=8),
                               activation=hp.Choice(f'activation_{i+2}', values=['relu', 'tanh', 'sigmoid'])))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd']),
                  loss='binary_crossentropy',
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      balanced_accuracy
                    #   BalancedAccuracy(name='balanced_accuracy', mask=mask)
                      
                      ] #+ custom_scorers
        )

    return model

# ---------------------------------------------------------------------------------------------------------------------

def create_model(optimizer='adam', input_dim=1):
    
    # create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu' , input_dim=input_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

# ---------------------------------------------------------------------------------------------------------------------

def dnn_tuner(self, pipe, X_train, y_train, groups, sample_weight=None, cv=None, exclude_mask=None):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    
    #groups_kfold = GroupKFold(n_splits=cv)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.7) # TODO: use GroupKFold!!!
    train_idx, val_idx = next(gss.split(X_train, y_train['class'], groups))
    x_train_, x_val = X_train.reset_index(drop=False).rename(columns={'index':'old_idx'}).iloc[train_idx].set_index('old_idx'), X_train.reset_index(drop=True).iloc[val_idx]
    y_train_, y_val = y_train['class'].reset_index(drop=False).rename(columns={'index':'old_idx'}).iloc[train_idx].set_index('old_idx').astype('float32'), y_train['class'].reset_index(drop=True).iloc[val_idx].astype('float32')

    if exclude_mask is not None:
        ignore = y_val.index.isin(exclude_mask)
        x_val, y_val = x_val[~ignore], y_val[~ignore]

    #tf.expand_dims(X, axis=-1) -> https://github.com/mrdbourke/tensorflow-deep-learning/discussions/256
    x_train_ = tf.expand_dims(self.get_transformed_data(x_train_, fit=True), axis=-1)
    x_val = tf.expand_dims(self.get_transformed_data(x_val), axis=-1)

    global input_dim  # TODO: find another way
    input_dim = x_train_.shape[1]
    
    tuner = pipe['classifier']
    tuner.search(x_train_, y_train_,
                 epochs=10, # TODO: how much? 100?
                 validation_data=(x_val, y_val),
                 callbacks=[es])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train_, y_train_, epochs=5, validation_split=0.2)
    val_acc_per_epoch = history.history['val_balanced_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(x_train_, y_train_, epochs=best_epoch, validation_split=0.2)
    
    return hypermodel