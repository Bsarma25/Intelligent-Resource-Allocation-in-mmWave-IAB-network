import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model

# Learning rate for actor-critic models
critic_lr = 1e-4
actor_lr = 5e-5

# Creating Optimizer for actor and critic networks
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.95

# Used to update target networks
tau = 0.005

class Buffer:
    def __init__(self, dim_agent_state,dim_action_space,num_agents,buffer_capacity=30000, batch_size=128):

        
        self.dim_agent_state=dim_agent_state
        self.dim_action_space=dim_action_space
        self.num_agents=num_agents
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity,self.num_agents ,self.dim_agent_state))
        self.action_buffer = np.zeros((self.buffer_capacity,self.num_agents,self.dim_action_space))
        self.reward_buffer = np.zeros((self.buffer_capacity,self.num_agents))
        self.next_state_buffer = np.zeros((self.buffer_capacity,self.num_agents,self.dim_agent_state))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self, actor_models, critic_models, target_actor_models, target_critic_models):
    
      # Updating networks of all the agents
      # by looping over number of agent
      for i in range(self.num_agents):
      
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training  and Updating ***critic model*** of ith agent
        target_actions = np.zeros((self.batch_size, self.num_agents,self.dim_action_space))
        for j in range(self.num_agents):
            
            ## Passing each state into corresponding target model to get action.
            corresponding_state=next_state_batch[:,j]
            target_actions[:,j]=target_actor_models[j](corresponding_state)

            # target_actions[:,j] = tf.reshape( temp_target_action, [self.batch_size])

        target_action_batch1 = target_actions[:,0]
        target_action_batch2 = target_actions[:,1]
        action_batch1 = action_batch[:,0]
        action_batch2 = action_batch[:,1]
    
        # Finding Gradient of loss function
        critic_agent_state_size=next_state_batch.shape[1]*next_state_batch.shape[2]
        flattened_state_batch=tf.reshape(state_batch, [-1,  critic_agent_state_size])
        flattened_next_state_batch=tf.reshape(next_state_batch, [-1,  critic_agent_state_size])
        
        with tf.GradientTape() as tape:
            y = reward_batch[:,i] + gamma * target_critic_models[i]([
                                                          flattened_next_state_batch, target_action_batch1, 
                                                          target_action_batch2])
            
            critic_value = critic_models[i]([
                                         flattened_state_batch, action_batch1, action_batch2])
            
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_models[i].trainable_variables)
        
        # Applying gradients to update critic network of ith agent
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_models[i].trainable_variables)
        )
        # Updating and training of ***critic network*** ended

    
    
    
        
        
        
        
        # Updating and Training of ***actor network** for ith agent
        
        
        actions = np.zeros((self.batch_size, self.num_agents,self.dim_action_space))
        for j in range(self.num_agents):
            a = actor_models[j](state_batch[:,j])
            actions[:,j]=a
            # actions[:,j] = tf.reshape(a, [self.batch_size])

        # Finding gradient of actor model if it is 1st agent
        if i == 0:
          
            with tf.GradientTape(persistent=True) as tape:
                # print('statebatch[0]',np.array([state_batch[:,i][0]]))
                action_ = actor_models[i]([state_batch[:,i]])

                critic_value = critic_models[i]([flattened_state_batch, action_, actions[:,1]])
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_models[i].trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad, actor_models[i].trainable_variables))

        
        # Finding gradient of actor model if it is 2nd agent
        elif i == 1:
            with tf.GradientTape(persistent=True) as tape:

                action_ = actor_models[i]([state_batch[:,i]])

                critic_value = critic_models[i]([flattened_state_batch, actions[:,0],action_])
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor_models[i].trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grad, actor_models[i].trainable_variables))

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau, ac_models, cr_models, target_ac, target_cr):

    for i in range(2):

        new_weights = []
        target_variables = target_cr[i].weights

        for j, variable in enumerate(cr_models[i].weights):
            new_weights.append(variable * tau + target_variables[j] * (1 - tau))

        target_cr[i].set_weights(new_weights)

        new_weights = []
        target_variables = target_ac[i].weights

        for j, variable in enumerate(ac_models[i].weights):
            new_weights.append(variable * tau + target_variables[j] * (1 - tau))

        target_ac[i].set_weights(new_weights)
