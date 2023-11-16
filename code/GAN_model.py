import tensorflow as tf
from keras import layers
import numpy as np

class GAN:
    def __init__(self, input_shape):
        self.input_shape = input_shape  # Input shape should be the shape of your financial bar data

    def build_generator(self):
        """
        Build the generator network.
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(256, input_shape=(self.input_shape,)))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(layers.Reshape(self.input_shape))
        return model

    def build_discriminator(self):
        """
        Build the discriminator network.
        """
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=self.input_shape))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (real or fake)
        return model

    def compile(self, optimizer='adam'):
        """
        Compile the GAN model.
        
        Args:
        optimizer: The optimizer to use. Example: 'adam'
        """
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates bars
        z = layers.Input(shape=(self.input_shape,))
        bar = self.generator(z)

        # For the combined model, we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated bars as input and determines validity
        valid = self.discriminator(bar)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = tf.keras.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


      #  print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss}]")

    def train(self, epochs, batch_size, sample_interval):
    
    # Load and prepare real data
    # ...
    
        half_batch = batch_size // 2

        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of real images
            idx = np.random.randint(0, real_data.shape[0], half_batch)
            real_imgs = real_data[idx]
            
            # Generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, self.noise_shape))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.noise_shape))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # If at save interval => save generated image samples and/or save model checkpoints
            if epoch % sample_interval == 0:
                # Output training progress
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")