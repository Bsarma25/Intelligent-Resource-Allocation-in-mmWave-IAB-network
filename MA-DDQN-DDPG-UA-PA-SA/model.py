import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model



def get_actor(state_size,action_size):
    
    inputs = layers.Input(shape=(state_size))
    out = layers.Dense(512, activation="relu", kernel_initializer="lecun_normal")(inputs)
    out = layers.Dense(512, activation="relu", kernel_initializer="lecun_normal")(out)
    action = layers.Dense(128, activation="relu", kernel_initializer="lecun_normal")(out)
    outputs = layers.Dense(int(action_size),activation='softmax')(action)    
    model = tf.keras.Model(inputs, outputs)
    return model





def get_critic(all_state_size,action_size):

    state_input = layers.Input(shape=(all_state_size))
    state_out = layers.Dense(128, activation="relu", kernel_initializer="lecun_normal")(state_input)
    state_out = layers.Dense(256, activation="relu", kernel_initializer="lecun_normal")(state_out)

    # Action all the agents as input
    action_input1 = layers.Input(shape=(action_size))
    action_input2 = layers.Input(shape=(action_size))
    action_input = layers.Concatenate()([action_input1, action_input2])
    action_out = layers.Dense(512, activation="relu", kernel_initializer="lecun_normal")(action_input)
    
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(512, activation="relu", kernel_initializer="lecun_normal")(concat)
    out = layers.Dense(512, activation="relu", kernel_initializer="lecun_normal")(out)
    outputs = layers.Dense(1)(out)
    model = tf.keras.Model([state_input, action_input1, action_input2], outputs)

    return model

# Executing Policy using actor models
def policy(state, noise_object, model,train='train'):
    
    sampled_actions = tf.squeeze(model(state))
    
    # Adding noise to action
    if train=='train':
        noise = noise_object()
        sampled_actions = sampled_actions.numpy() + noise
        
    elif train=='predict':
        sampled_actions = sampled_actions.numpy()     

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, 0, 1)

    return [np.squeeze(legal_action)]

# Executing Policy using actor models
def train_policy(state, noise_object, model,k,epsilon):
    
    sampled_actions = tf.squeeze(model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    
    # if np.random.random() < epsilon: # Decay epsilon from 1 to 0.1
    #     power_action=np.random.random(3)
    #     sampled_actions=power_action # last 3 actions are channel, 1st 3 are power

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, 0, 1)

    return [np.squeeze(legal_action)]