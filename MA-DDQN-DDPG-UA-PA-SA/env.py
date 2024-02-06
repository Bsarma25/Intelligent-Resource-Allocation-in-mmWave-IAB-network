import numpy as np
import matplotlib.pyplot as plt
from random import random, uniform, choice, randrange, sample
import itertools
# import torch

users_IAB_1=[1,2,3]
users_IAB_2=[4,5,6]

class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))
    
    
class Nodes:  # Define the base station

    def __init__(self, sce, BS_index, BS_type, BS_Loc, BS_Radius):
        self.sce= sce
        self.id = BS_index # basestation index
        self.BStype = BS_type # DBS, IAB Node
        self.BS_Loc = BS_Loc #Location
        self.BS_Radius = BS_Radius # Range

    def Get_Location(self):
        return self.BS_Loc

class UE:  #define a UE

   def __init__(self, sce, UE_index, UE_Loc):
        self.sce= sce
        self.id = UE_index # UE index
        self.UE_Loc = UE_Loc #Location


   def Get_Location(self):
        return self.UE_Loc

   def Set_Location(self, new_location):
        self.UE_Loc = new_location

def calculate_decay_rate(decay_init, decay_final, num_episodes):
    return (decay_final / decay_init) ** (1 / num_episodes)

def decay_exploration(episode, decay_rate):
    return max(decay_final, decay_init * decay_rate ** episode)

def plot_locations(scenario,sce,opt):


    ue_locations=[]
    bs_locations=[]
    for i in scenario.Get_UEs():
      ue_locations.append(i.Get_Location())
    for i in scenario.Get_BaseStations():
      bs_locations.append(i.Get_Location())

    fig, ax = plt.subplots()
    for i in np.arange(3):
      ax.scatter(bs_locations[i][0],bs_locations[i][1],c='b')
      ax.text(bs_locations[i][0], bs_locations[i][1], f'BS{i}', fontsize=12, ha='left', va='bottom')

      if i==0:
        dbs_circle=plt.Circle( (bs_locations[i][0],bs_locations[i][1]),sce.rDBS,alpha=0.1)
        # note we must use plt.subplots, not plt.subplot
        ax.add_patch(dbs_circle)

      else:
        iab_circle=plt.Circle( (bs_locations[i][0],bs_locations[i][1]),sce.rIAB,alpha=0.1)
        ax.add_patch(iab_circle)

    for i in np.arange(sce.nUE):
      ax.scatter(ue_locations[i][0],ue_locations[i][1],c='g')
      ax.text(ue_locations[i][0], ue_locations[i][1], f'UE{i}', fontsize=12, ha='left', va='bottom')

    plt.show()

class Scenario:  # Define the network scenario

    def __init__(self, sce):  # Initialize the scenario we simulate
        self.sce = sce
        self.BaseStations = self.BS_Init()
        self.UEs =   self.UE_Init()



    def reset(self):   # Reset the scenario we simulate
        for i in range(len(self.BaseStations)):
            self.BaseStations[i].reset()

    def BS_Number(self):
        nBS = self.sce.nDBS + self.sce.nIAB # The number of base stations
        return nBS

    def BS_Location(self):
        Loc_DBS =  np.zeros((1,2)) # Initialize the locations of BSs
        Loc_IAB = np.zeros((self.sce.nIAB,2))

        for i in range(self.sce.nIAB):
            Loc_IAB[i,0] = Loc_DBS[0,0] + self.sce.dist_IAB_centre*np.cos(np.pi/2*(i%4))
            Loc_IAB[i,1] = Loc_DBS[0,1] + self.sce.dist_IAB_centre*np.sin(np.pi/2*(i%4))

        return Loc_DBS, Loc_IAB

    def UE_Location(self):
        Loc_DBS,Loc_IAB= self.BS_Location() # Initialize the locations of BSs
        Loc_UEs = np.zeros((self.sce.nUE,2))

        r = self.sce.rDBS * random()
        theta = uniform(-np.pi, np.pi)
        Loc_UEs[0, 0] = Loc_DBS[0, 0] + r * np.cos(theta)
        Loc_UEs[0, 1] = Loc_DBS[0, 1] + r * np.sin(theta)
        r_min=5
        ue_count=1
        for i in range(self.sce.nIAB):
          for j in range(ue_count,ue_count+self.sce.K):
            r=self.sce.rIAB*random()+r_min
            theta=uniform(-np.pi,np.pi)
            Loc_UEs[j, 0] = Loc_IAB[i, 0] + r * np.cos(theta)
            Loc_UEs[j, 1] = Loc_IAB[i, 1] + r * np.sin(theta)
            ue_count+=1
        return Loc_UEs


    def BS_Init(self):   # Initialize all the base stations
        BaseStations = []  # The vector of base stations
        Loc_DBS, Loc_IAB = self.BS_Location()


        for i in range(self.sce.nDBS):  # Initialize the MBSs
            BS_index = i
            BS_type = "DBS"
            BS_Loc = Loc_DBS[i]
            BS_Radius = self.sce.rDBS
            BaseStations.append(Nodes(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))

        for i in range(self.sce.nIAB):
            BS_index = self.sce.nDBS + i
            BS_type = "IAB"
            BS_Loc = Loc_IAB[i]
            BS_Radius = self.sce.rIAB
            BaseStations.append(Nodes(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))


        return BaseStations

    def UE_Init(self):   # Initialize all the UEs
        UEs = []  # The vector of UEs
        Loc_UE= self.UE_Location()
        for i in range(self.sce.nUE):
            UE_index = i
            UE_Loc = Loc_UE[i]
            UEs.append(UE(self.sce, UE_index,UE_Loc))

        return UEs


    def Get_BaseStations(self):
        return self.BaseStations

    def Get_UEs(self):
        return self.UEs


    def UEs_move(self,step_size=1):

        direction_vectors = np.random.uniform(-1, 1, (self.sce.nUE, 2))

        # Normalize the direction vectors
        norms = np.linalg.norm(direction_vectors, axis=1)
        direction_vectors = direction_vectors / norms[:, np.newaxis]
        for i, ue in enumerate(self.UEs):
            # Get current UE location
            Loc_UE = ue.Get_Location()

            # Calculate the new position

            #(0,0), Max radius=200, +1*direction
            new_UE_location = Loc_UE + step_size * direction_vectors[i]

            # # Check if the UE has moved outside the coverage area of the BS
            distance = np.linalg.norm(new_UE_location)
            if distance > self.sce.rIAB:
            #     # If the UE has moved outside the coverage area, move it back to the boundary
                 new_UE_location = new_UE_location / distance * self.sce.rIAB

            # # Update the location
            ue.Set_Location(new_UE_location)

        return np.array([ue.Get_Location() for ue in self.UEs])
    
    
class Processing:
    def __init__(self):
        pass

    def distance_matrix_int(self,scenario):
      BS=scenario.Get_BaseStations()
      UEs=scenario.Get_UEs()
      distance_matrix=np.zeros((len(BS),len(UEs)))  #distance of all  users from all agents(BSs)
      distance_IAB2IAB=np.zeros((len(BS)-1,len(BS)-1))

      for i in range(len(BS)):
        for j in range(len(UEs)):
              loc_diff=UEs[j].Get_Location() - BS[i].Get_Location()
              distance_matrix[i][j] = np.sqrt((loc_diff[0]**2 + loc_diff[1]**2))

      for i in range(len(BS)-1):
        for j in range(len(BS)-1):
              loc_diff=BS[j+1].Get_Location() - BS[i+1].Get_Location()
              distance_IAB2IAB[i][j] = np.sqrt((loc_diff[0]**2 + loc_diff[1]**2))
      return distance_matrix ,distance_IAB2IAB




    def distance_list(self,scenario,index):
      distance=[]
      UEs=scenario.Get_UEs()
      BS=scenario.Get_BaseStations()
      for i in range(len(UEs)):
                Loc_diff = UEs[i].Get_Location() - BS[index].Get_Location
                dist = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))  # Calculate the distance between BS and UE
                distance.append(dist)
      return distance


class Agent_IAB:

    def __init__(self, opt, sce,action_size,state_size,scenario,Agent_type, index, device):  # Initialize the agent (UE)
        self.opt = opt # Optimizer varaibles
        self.sce = sce # Scenario variables
        self.id = index
        self.agent_type=Agent_type
        self.device = device
        self.action_size =action_size#self.sce.K
        self.distance_UEs_this_IAB=self.sce.nUE
        self.normalized_Rate_size=self.sce.K
        self.state_size=state_size
        self.L = scenario.BS_Number()
        # self.h=np.random.exponential(scale=1.0)
        self.current_h=[None for _ in range(self.sce.K)]  #new
        self.normalized_SINR=None
        self.normalized_rate=None
        self.normalized_threshold=None
        self.ch_gain=[None]*self.sce.K
        self.max_ch=sce.nChannel
        print('State size of agent',index,':',self.state_size)
        print('Action size of agent',index,':',self.action_size)
        self.location = self.Set_Location(scenario)
        self.dist_matrix1,_=self.distance_matrix_int(scenario)

    def Set_Location(self, scenario):  # Initialize the location of the agent
        _,Loc_IAB = scenario.BS_Location()

        LocM= Loc_IAB[self.id]
        LocM=np.reshape(LocM,(-1,2))
        Loc_agent = np.zeros((1,2))
        Loc_agent[0,0] = LocM[0,0]
        Loc_agent[0,1] = LocM[0,1]
        return Loc_agent[0]

    def Get_Location(self):
        return self.location

    def pathloss(self,r):  #r is the distance

        p_los = np.exp(-self.sce.beta*r)  # LOS probability
        p_nlos = 1 - p_los  # NLOS probability
        # path loss
        # L_sig=
        L = p_los * self.sce.AL * r**(-self.sce.alphaL) + p_nlos * self.sce.ANL * r**(-self.sce.alphaNL)
        L_sig= self.sce.AL * r**(-self.sce.alphaL)
        # p = np.random.binomial(1, p_los)
        # if(p==1):
        #   L_inter=AL * r**(-alphaL)
        # else:
        L_inter= self.sce.ANL * r**(-self.sce.alphaNL)

        return L_sig,L_inter

    def gain(self):


      # Convert to linear scale

        M = 10**(self.sce.M_db / 10)
        m= 10**(self.sce.m_db / 10)

      # antenna gain probabilities
        Phi = np.radians(self.sce.Phi_rad)
        p_MM = Phi**2 / (4 * np.pi**2)
        p_Mm = Phi * (2*np.pi - Phi) / (2 * np.pi**2)
        p_mm = (2*np.pi - Phi)**2 / (4 * np.pi**2)

        # antenna gain
        Gain_array=np.array([M*M,M*m,m*m])
        Gain_probs=np.array([p_MM,p_Mm,p_mm])
        return Gain_array,Gain_probs

    # def rcv_power(self,distance,P_t):
    #     Total_Tx_Power_dBm=33
    #     Tx_Power=(10 ** (( Total_Tx_Power_dBm- 30) / 10))*P_t
    #     Gain,_=self.gain()
    #     L,_=self.pathloss(distance)
    #     # h=np.random.exponential(scale=1.0)
    #     P_rcv =  Tx_Power * Gain[0] *L*self.h
    #     return P_rcv
    def rcv_power(self,distance,P_t,h=None):
         
        Total_Tx_Power_dBm=33
        Tx_Power=(10 ** (( Total_Tx_Power_dBm- 30) / 10))*P_t
        Gain,_=self.gain()
        L,_=self.pathloss(distance)
        if h is None:
            h=np.random.exponential(scale=1.0)
        # h=np.random.exponential(scale=1.0)
        h_= Gain[0] *L
        P_rcv =  Tx_Power * Gain[0] *L
        return P_rcv,h,h_

    def interference_power(self,distance,P_t):
       
        Total_Tx_Power_dBm=33
        Tx_Power=(10 ** (( Total_Tx_Power_dBm- 30) / 10))*P_t
        Gain,probs=self.gain()
        _,L_inter=self.pathloss(distance)
        interference=0
        for i in [0,1,2]:
          h1=np.random.exponential(scale=1.0)
          interference= interference+Tx_Power*probs[i]*Gain[i]*L_inter
        return interference

    def interfernce_from_IAB(self, action_IAB, id_other_agent):
        index = self.id
        subchannel_selected_matrix = action_IAB[:, -self.sce.K:]
        subchannel_selected_matrix=self.scale_channel_values(subchannel_selected_matrix)
        subchannel_selected = subchannel_selected_matrix[self.id]
        users_IAB = users_IAB_1 if self.id == 0 else users_IAB_2
        subchannel_selected_other = subchannel_selected_matrix[id_other_agent]
        action_IAB_other = action_IAB[id_other_agent]
        power_IAB_other = action_IAB_other[:self.sce.K]

        # Create a dictionary with subchannels as keys and list of powers as values
        subchannel_power_dict_other = {}
        for subchannel, power in zip(subchannel_selected_other, power_IAB_other):
            if subchannel in subchannel_power_dict_other:
                subchannel_power_dict_other[subchannel].append(power)
            else:
                subchannel_power_dict_other[subchannel] = [power]

        interference_IAB_UE = np.zeros(len(users_IAB))
        distance_matrix = self.dist_matrix1[id_other_agent + 1, :]

        for i, user in enumerate(users_IAB):
            subchannel_selected_i = subchannel_selected[i]

            if subchannel_selected_i in subchannel_power_dict_other:
                for power in subchannel_power_dict_other[subchannel_selected_i]:

                    Rcv_interference_power = self.interference_power(distance_matrix[i], power)
                    interference_IAB_UE[i] += Rcv_interference_power

        return interference_IAB_UE


    def total_interference(self, action_IAB):
        # interference_DBS = self.interfernce_from_DBS(action_IAB, action_DBS)  # Interference from DBS
        interference_total = 0#interference_DBS.copy()  # Initialize total interference with DBS interference

        for agent_id in range(self.sce.nIAB):  # Replace n_agents with the total number of agents
             if agent_id != self.id:
                interference_IAB = self.interfernce_from_IAB(action_IAB, agent_id)  # Interference from other IAB agents
                interference_total += interference_IAB

        return interference_total  #


    def scale_channel_values(self,subchannel_unscaled):

      subchannel=subchannel_unscaled.copy()  #last K sigmoid values
      subchannel = np.multiply(subchannel, self.max_ch)
      subchannel = np.ceil(subchannel)
      subchannel[subchannel == 0]=1
      return subchannel


    def Get_Reward_IAB(self,scenario,action_IAB):   #Calculate RAte of DBS to IAB links

        UEs=scenario.Get_UEs()
        action=action_IAB[self.id]
        
        # subchannel_selected=action[-self.sce.K:]
        # subchannel_selected=self.scale_channel_values(subchannel_selected)
        # power=action[:self.sce.K]
        power=action
        subchannel_selected=[1,1,1]
        
        action1=np.concatenate([action_IAB[0],[1,1,1]])
        action2=np.concatenate([action_IAB[1],[1,1,1]])
        action_IAB_full=np.array((action1,action2))

        if self.id==0:

          users=np.array(users_IAB_1)
        else:
            

          users=np.array(users_IAB_2)

        Rx_power=np.zeros(len(users))
        SINR= np.zeros(len(users))
        distance=np.zeros(len(users))
        Rate= np.zeros(len(users))  #storing the rates of the associated users
        reward= np.zeros(len(users))  #storing the rewards from each associated reward.total reward by an IAB agent is the sum of all users rewards
        state= np.zeros(len(users))
        #Finding Noise, SINR_max, Rate_max -> Single value
        noise_dbm = self.sce.N0+ 10 * np.log10(self.sce.BW)
        Noise=10 ** ((noise_dbm - 30) / 10)
        cost=0
        penalty=0
        profit=0
        ch_gain=np.zeros(len(users)) 
        SINR_max= (10 ** (( 33- 30) / 10))/Noise
        Rate_max=np.log2(1+SINR_max)*(250/self.sce.nChannel)############remove the scaling and run it
        # print(action_IAB_full)
        interference_other_agents=self.total_interference(action_IAB_full)

        for count,UE_selected in enumerate(users):

          Loc_diff= UEs[UE_selected].Get_Location()-self.location
          distance[count] = np.sqrt((Loc_diff[0]**2 + Loc_diff[1]**2))  # Calculate the distance between BS and UE
          # Rx_power[count] =self.rcv_power(distance[count],power[count])# Calculate the received power of each associated user
          if self.current_h[count] is None:
                Rx_power[count], self.current_h[count],self.ch_gain[count] = self.rcv_power(distance[count],power[count])
          else:
                Rx_power[count], _,self.ch_gain[count] =self.rcv_power(distance[count],power[count],h=self.current_h[count])

        int_noise=[]
        for i in range(len(users)):
            subchannel_allocation=subchannel_selected[i]
            interference_UE_UE=0.0  #interference from other UEs connected to the same BS

            for j in range(len(users)):
                if(j!=i):
                    subchannel_selected_j=subchannel_selected[j]
                    if(subchannel_selected_j==subchannel_allocation):
                      #calculate interfernce from this IAB-UE
                      cost=cost+penalty
                      Rcv_interference_power=self.interference_power(distance[i],power[j]) #new 
                      # Rcv_interference_power=self.rcv_power(distance[i],power[j])
                      interference_UE_UE=interference_UE_UE+Rcv_interference_power#new
                      # print('Interference: ',i,j)
                      # print(subchannel_allocation,subchannel_selected_j,interference_UE_UE)
                      # print('\n')


            # interference_UE_UE = interference_UE_UE-Rx_power[i]
            interference=interference_UE_UE*10+interference_other_agents[i]

            SINR[i] = Rx_power[i]/(interference + Noise)
            Rate[i] = np.log2(1+SINR[i])*(250/self.sce.nChannel)

        self.normalized_SINR=np.divide(SINR,SINR_max)
        self.normalized_rate=np.divide(Rate,(Rate_max/10))
        self.normalized_threshold=(10**(self.sce.QoS_thr/10))/Rate_max

        min_rate=0.5*(250/self.sce.nChannel)
        if min(Rate)< min_rate:
            Reward=-5
        else:
            reward_=np.sum(Rate)
            Reward =reward_/100
        # print('reward',Reward)                   
        return Reward,Rate,SINR,Rx_power,int_noise,state
    
    def distance_matrix_int(self,scenario):
        BS=scenario.Get_BaseStations()
        UEs=scenario.Get_UEs()
        distance_matrix=np.zeros((len(BS),len(UEs)))  #distance of all  users from all agents(BSs)
        distance_IAB2IAB=np.zeros((len(BS)-1,len(BS)-1))

        for i in range(len(BS)):
            for j in range(len(UEs)):
                loc_diff=UEs[j].Get_Location() - BS[i].Get_Location()
                distance_matrix[i][j] = np.sqrt((loc_diff[0]**2 + loc_diff[1]**2))

        for i in range(len(BS)-1):
            for j in range(len(BS)-1):
                loc_diff=BS[j+1].Get_Location() - BS[i+1].Get_Location()
                distance_IAB2IAB[i][j] = np.sqrt((loc_diff[0]**2 + loc_diff[1]**2))
        return distance_matrix ,distance_IAB2IAB


    def get_state_IAB(self,scenario,prev_action_IAB):
        prev_action_IAB_copy=prev_action_IAB.copy()
        distance_matrix,_=self.distance_matrix_int(scenario)
        s1=self.normalized_rate
        s2=self.ch_gain
        if self.id==0:
          users=np.array(users_IAB_1)
        else:
          users=np.array(users_IAB_2)
        distance_=distance_matrix[self.id+1,:]
        distance=distance_[users]  #distance of the connected users
       
        pathloss_=np.zeros(len(users))
        r_min=5
        max_pl_= self.sce.AL * r_min**(-self.sce.alphaL)
        # max_pl=np.log10(max_pl_)
        for i in range(len(users)):
           pathloss_[i],_=self.pathloss(distance[i])
           # pathloss_[i]=np.log10(pathloss_[i])/max_pl  #normalized
          
        state=np.concatenate([s1,distance/self.sce.rIAB,prev_action_IAB_copy[self.id].flatten()])
        #  print('state: ',state)
        return state

    def initial_action_IAB(self):

      #random_power_allocation
      action= np.random.random(self.sce.K)#*self.sce.nIAB)
      # power /= power.sum()
      # subchannel=np.random.randint(low=1, high=self.sce.nChannel, size=self.sce.K)
      # action=np.concatenate([power,subchannel])
      return action

def discretized_float_channelselected(action):

    # Optimal
    discrete_actions=action[-sce.K:].copy()
    range_gap=1/sce.nChannel
    ranges=np.arange(0,1,range_gap)
    ranges=np.append(ranges,1)
    for index in np.arange(len(discrete_actions)):
      j=discrete_actions[index]
      for i in np.arange(len(ranges[:-1])):
        if ((j>=ranges[i]) & (j<=ranges[i+1])):
          k=np.mean([ranges[i],ranges[i+1]])
          discrete_actions[index]=k
    return np.concatenate([action[:sce.K],discrete_actions])

def normalize_power(action):

    power_values = action.copy()
    power_values = power_values / (np.sum(power_values)+1e-8)

    # channel_selection=action[-sce.K:]
    return power_values #np.concatenate([power_values])#,channel_selection])

def get_channel_action_space():
    x = np.arange(1,sce.nChannel+1)
    action_space=[p for p in itertools.product(x, repeat=sce.K)]
    return action_space

def merge_actions(continuous_action,channel_action):
    return np.concatenate([continuous_action, channel_action])

def initial_state(scenario,Agent,prev_state,sce,dim_agent_action):
    action_IAB=np.zeros([sce.nIAB,dim_agent_action])
    for i in range(2):
        action_IAB[i]=np.array(Agent[i].initial_action_IAB())

    for i in range(2):
        _,_,_,_,_,_= Agent[i].Get_Reward_IAB(scenario,action_IAB)
        prev_state[i]=Agent[i].get_state_IAB(scenario,action_IAB) # has to be run after get reward
    
    return prev_state

def step(scenario,Agent,action_IAB):

    reward=[]
    new_state=[]
    for i in range(2):
       
        reward_i,Rate,SINR,_,_,_  = Agent[i].Get_Reward_IAB(scenario,action_IAB)
        reward.append(reward_i)
        new_state.append(Agent[i].get_state_IAB(scenario,action_IAB))
    
    return new_state,reward