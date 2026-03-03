import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

LATENT_DIM=100; N_CLASSES=2; N_FEATURES=4

def build_generator(ld, nc, nf):
    noise=layers.Input(shape=(ld,))
    label=layers.Input(shape=(1,),dtype='int32')
    emb=layers.Flatten()(layers.Embedding(nc,16)(label))
    x=layers.Concatenate()([noise,emb])
    x=layers.Dense(128,activation='relu')(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dense(256,activation='relu')(x)
    x=layers.BatchNormalization()(x)
    out=layers.Dense(nf,activation='tanh')(x)
    return models.Model([noise,label],out)

gen = build_generator(LATENT_DIM, N_CLASSES, N_FEATURES)

def generate_fraud(gen, n=1000):
    noise=np.random.normal(0,1,(n,LATENT_DIM))
    labels=np.ones((n,1),dtype=int)  # fraud class = 1
    return gen.predict([noise,labels])

synthetic_fraud = generate_fraud(gen, 1000)
print(f'Generated {len(synthetic_fraud)} synthetic fraud records')
