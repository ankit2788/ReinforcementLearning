from tensorflow import keras
from tensorflow.keras.layers import Dense
from copy import copy


class Dummy():
    def __init__(self):
        pass

class ActorCritic_FF(keras.Model):

    # plain feed forward actor critic model    

    def __init__(self, state_size, action_size, actorHiddenUnits = [32], criticHiddenUnits = [32]):

        super().__init__()

        self.state_size = state_size
        self.action_size =  action_size

        self.actorHiddenUnits   = actorHiddenUnits
        self.criticHiddenUnits  = criticHiddenUnits

        # create 2 separate models for actor and critic
        self.actor = Dummy()
        self.critic = Dummy()

        # --- Actor --> returns prob distribution for all possible states given state
        for index, layer in enumerate(actorHiddenUnits):
            layer = Dense(units = layer, activation="relu")
            setattr(self.actor, f"layer{index}", layer)

        self.actor.policy = Dense(units = self.action_size, activation="softmax")
        
        # --- Critic   --> Value function (given a state)
        for index, layer in enumerate(criticHiddenUnits):
            layer = Dense(units = layer, activation="relu")
            setattr(self.critic, f"layer{index}", layer)
        self.critic.values = Dense(units = 1, activation="linear")



    def call(self, inputs):

        # simple forward pass
        actor_inputs = copy(inputs)
        critic_inputs = copy(inputs)


        # Actor
        for index, layer in enumerate(self.actorHiddenUnits):
            actor_inputs = getattr(self.actor, f"layer{index}")(actor_inputs)
        
        probs = self.actor.policy(actor_inputs)

        # Critic
        for index, layer in enumerate(self.criticHiddenUnits):
            critic_inputs = getattr(self.critic, f"layer{index}")(critic_inputs)

        values = self.actor.values(critic_inputs)

        return probs, values



