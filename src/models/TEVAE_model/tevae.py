"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands

Original paper DOI: arXiv:2407.06849
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions


class TeVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, ma, beta=1e0):
        super(TeVAE, self).__init__()

        # Model
        self.encoder = encoder
        self.decoder = decoder
        self.ma = ma

        # Metrics
        self.beta_tracker = tf.keras.metrics.Mean(name="beta")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.negloglik_loss_tracker = tf.keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        # Modifiable weight for KL-loss
        self.beta = tf.Variable(beta, trainable=False)  # Weight for KL-Loss, can be modified with a callback

    @tf.function
    def loss_fn(self, X, Xhat, Xhat_mean, Xhat_logvar, Z_mean, Z_logvar):
        # Configure distribution with parameters output from decoder
        output_dist = tfd.MultivariateNormalDiag(loc=Xhat_mean, scale_diag=tf.sqrt(tf.math.exp(Xhat_logvar)))
        # Calculate log probability of sample belongs parametrised distribution
        loglik_loss = output_dist.log_prob(X)

        # Calculate KL Divergence between latent distribution and Gaussian distribution
        kl_loss = tfd.kl_divergence(
            tfd.MultivariateNormalDiag(loc=Z_mean, scale_diag=tf.sqrt(tf.math.exp(Z_logvar))),
            tfd.MultivariateNormalDiag(loc=tf.zeros_like(Z_mean), scale_diag=tf.ones_like(Z_logvar))
        )

        return -tf.reduce_sum(loglik_loss, axis=1), tf.reduce_sum(kl_loss, axis=1)

    # @tf.function
    def train_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            Z_mean, Z_logvar, Z, states = self.encoder(X, training=True)
            # Forward pass through MA
            C = self.ma([X, Z], training=True)
            # Forward pass through decoder
            Xhat_mean, Xhat_logvar, Xhat = self.decoder(C, training=True)
            # Calculate losses from parameters
            negloglik_loss, kl_loss = self.loss_fn(
                X,
                Xhat,
                Xhat_mean,
                Xhat_logvar,
                Z_mean,
                Z_logvar
            )
            # Calculate total loss from different losses
            total_loss = negloglik_loss + self.beta * kl_loss
        # Calculate gradients in backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Track losses
        self.beta_tracker.update_state(self.beta)
        self.total_loss_tracker.update_state(total_loss)
        self.negloglik_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "beta": self.beta_tracker.result(),
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.negloglik_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # @tf.function
    def test_step(self, X):
        if isinstance(X, tuple):
            X = X[0]
        # Forward pass through encoder
        Z_mean, Z_logvar, Z, states = self.encoder(X, training=False)
        # Forward pass through MA
        C = self.ma([X, Z_mean], training=False)
        # Forward pass through decoder
        Xhat_mean, Xhat_logvar, Xhat = self.decoder(C, training=False)
        # Calculate losses from parameters
        negloglik_loss, kl_loss = self.loss_fn(
            X,
            Xhat,
            Xhat_mean,
            Xhat_logvar,
            Z_mean,
            Z_logvar
        )
        total_loss = negloglik_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.negloglik_loss_tracker.update_state(negloglik_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics if m.name == 'rec_loss'}

    @property
    def metrics(self):
        return [
            self.beta_tracker,
            self.total_loss_tracker,
            self.negloglik_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        Z_mean, Z_logvar, Z, states = self.encoder(inputs, training=False)
        C = self.ma([inputs, Z_mean], training=False)
        Xhat_mean, Xhat_logvar, Xhat = self.decoder(C, training=False)
        return Xhat_mean, Xhat_logvar, Xhat, Z_mean, Z_logvar, Z, C


class TeVAE_Encoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, hidden_units, seed):
        super(TeVAE_Encoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.encoder = self.build_encoder()

    def build_encoder(self):
        enc_input = tfkl.Input(shape=(self.seq_len, self.features))
        enc_input = tfkl.GaussianNoise(0.01, seed=self.seed)(enc_input)
        bilstm = tfkl.Bidirectional(tfkl.LSTM(self.hidden_units*2, return_sequences=True))(enc_input)
        bilstm = tfkl.Bidirectional(tfkl.LSTM(self.hidden_units, return_sequences=True))(bilstm)
        Z_mean = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="Z_mean"))(bilstm)
        Z_logvar = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, name="Z_logvar"))(bilstm)
        output_dist = tfd.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(Z_mean), seed=self.seed)
        Z = Z_mean + tf.sqrt(tf.math.exp(Z_logvar)) * eps
        return tf.keras.Model(enc_input, [Z_mean, Z_logvar, Z, bilstm], name="encoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.encoder(inputs, **kwargs)


class TeVAE_Decoder(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, features, hidden_units, seed):
        super(TeVAE_Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.features = features
        self.hidden_units = hidden_units
        self.seed = seed
        self.decoder = self.build_decoder()

    def build_decoder(self):
        attention_input = tfkl.Input(shape=(self.seq_len, self.latent_dim))
        bilstm = tfkl.Bidirectional(tfkl.LSTM(self.hidden_units, return_sequences=True))(attention_input)
        bilstm = tfkl.Bidirectional(tfkl.LSTM(self.hidden_units*2, return_sequences=True))(bilstm)
        Xhat_mean = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_mean")(bilstm)
        Xhat_logvar = tfkl.TimeDistributed(tfkl.Dense(self.features), name="Xhat_logvar")(bilstm)
        output_dist = tfd.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(Xhat_mean), seed=self.seed)
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_logvar)) * eps
        return tf.keras.Model(attention_input, [Xhat_mean, Xhat_logvar, Xhat], name="decoder")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.decoder(inputs, **kwargs)


class MA(tf.keras.Model):
    def __init__(self, seq_len, latent_dim, key_dim, features):
        super(MA, self).__init__()

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.ma = self.build_MA()

    def build_MA(self):
        attention = tfkl.MultiHeadAttention(
            num_heads=8,
            key_dim=self.key_dim,
            output_shape=self.latent_dim,
            name="C"
        )

        raw_input = tfkl.Input(shape=(self.seq_len, self.features))
        latent_input = tfkl.Input(shape=(self.seq_len, self.latent_dim))
        C = attention(query=raw_input, key=raw_input, value=latent_input)
        return tf.keras.Model([raw_input, latent_input], C, name="MA")

    @tf.function
    def call(self, inputs, **kwargs):
        return self.ma(inputs, **kwargs)


if __name__ == "__main__":
    pass
