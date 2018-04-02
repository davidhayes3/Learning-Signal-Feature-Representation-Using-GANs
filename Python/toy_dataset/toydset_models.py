from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, BatchNormalization

# Settings
latent_dim = 2
img_dim = 4


# Define models

def encoder_model():
    model = Sequential()

    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    return model


def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(img_dim))
    model.add(Activation('tanh'))

    return model

def discriminator_model():
    z = Input(shape=(2,))
    z1 = Dense(512)(z)
    z2 = LeakyReLU(alpha=0.2)(z1)
    #z2 = Dense(512)(z1)

    x = Input(shape=(4,))
    x1 = Dense(512)(x)
    x2 = LeakyReLU(alpha=0.2)(x1)
    #x2 = Dense(512)(x1)

    #zx = merge([z2, x2], mode='dot', dot_axes=(1, 1))
    zx = concatenate([z2, x2])
    zx = Dense(1)(zx)
    validity = Activation('sigmoid')(zx)

    return Model([z, x], validity)


'''def discriminator_model():

    z = Input(shape=(latent_dim,))
    x = Input(shape=(img_dim,))
    d_in = concatenate([z, x])

    model = Dense(512)(d_in)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation="sigmoid")(model)

    return Model([z, x], validity)'''
