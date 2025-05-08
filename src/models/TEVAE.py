import numpy as np

from src.models.TEVAE_model import TeVAE, TeVAE_Encoder, TeVAE_Decoder, KL_annealing, MA
from src.templates.model import Model


class TEVAE(Model):
    """Implementation of TEVAE algorithm."""

    def __init__(self, observation_period, **kwargs):
        pass
        ######## BELOW WORKING CODE FROM NOTEBOOK TEVAE ########
        # for sample_nbr, ts_data, ts_label, ts_train, ts_test, ts_label_train, ts_label_test in dataset.get_ts(0.2):
        #     print(sample_nbr)
        #     print(ts_train.shape)
        #     break
        # model_save_path = 'models/TEVAE'
        # seed = 42
        # n_splits = 3  # Number of splits for cross-validation
        # tscv = TimeSeriesSplit(n_splits=n_splits)
        # window_size = 100  # Choose an appropriate window size for your time series
        # window_shift = window_size // 2
        # for split_idx, (train_idx, val_idx) in enumerate(tscv.split(ts_train)):
        #     # Split the data into training and validation sets
        #     train_data, val_data = [ts_train[train_idx]], [ts_train[val_idx]]
        #     print(train_data[0].shape)
        #     train_data = ts_processor.window_list(train_data, window_size, window_shift)
        #     val_data = ts_processor.window_list(val_data, window_size, window_shift)
        #     print(train_data.shape)
        #     print(val_data.shape)
        #     # Create TensorFlow datasets
        #     train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(1024).prefetch(tf.data.AUTOTUNE)
        #     val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(1024).prefetch(tf.data.AUTOTUNE)
        #     tfdata_train = tfdata_train.cache().batch(1024).prefetch(tf.data.AUTOTUNE)
        #     tfdata_val = tfdata_val.cache().batch(1024).prefetch(tf.data.AUTOTUNE)
        # 
        #     # Establish callbacks
        #     early_stopping = tf.keras.callbacks.EarlyStopping(
        #         monitor='val_rec_loss',
        #         mode='min',
        #         verbose=1,
        #         patience=250,
        #         restore_best_weights=True,
        #     )
        #     # KL Annealing
        #     annealing = KL_annealing(
        #         annealing_epochs=25,
        #         annealing_type="cyclical",
        #         grace_period=25,
        #         start=1e-3,
        #         end=1e-0,
        #     )
        #     # Define model
        #     window_size = tfdata_train.element_spec.shape[1]
        #     features = tfdata_train.element_spec.shape[2]
        #     latent_dim = features // 2
        #     key_dim = features // 8
        #     hidden_units = features * 16
        #     encoder = TeVAE_Encoder(seq_len=window_size, latent_dim=latent_dim, features=features, hidden_units=hidden_units, seed=seed)
        #     decoder = TeVAE_Decoder(seq_len=window_size, latent_dim=latent_dim, features=features, hidden_units=hidden_units, seed=seed)
        #     ma = MA(seq_len=window_size, latent_dim=latent_dim, key_dim=key_dim, features=features)
        #     model = TeVAE(encoder, decoder, ma)
        #     callback_list = [early_stopping, annealing]
        #     optimiser = tf.keras.optimizers.Adam(amsgrad=True)
        #     model.compile(optimizer=optimiser)
        #     # Fit vae model
        #     history = model.fit(tfdata_train,
        #                         epochs=1,
        #                         callbacks=callback_list,
        #                         validation_data=tfdata_val,
        #                         verbose=2
        #                         )
        #     # Run and save model
        #     model.predict(tf.random.normal((32, window_size, features)), verbose=0)
        #     model.save(model_save_path)
        #     with open(os.path.join(model_save_path, 'final_loss.txt'), 'x') as f:
        #         f.write(str(min(history.history['val_rec_loss'])))
        #     tf.keras.backend.clear_session()

    def train(self, **kwargs):
        pass

    def update(self, X: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        return score