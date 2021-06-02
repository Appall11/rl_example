import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module): # state + action
    def __init__(self, beta, input_dims, n_actions, name='critic', load_chkpt_dir='tmp/td3_load', save_chkpt_dir='tmp/td3_save'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.load_chkpt_dir = load_chkpt_dir
        self.load_chkpt_file = os.path.join(self.load_chkpt_dir, name+'_td3')
        self.save_chkpt_dir = save_chkpt_dir
        self.save_chkpt_file = os.path.join(self.save_chkpt_dir, name+'_td3')

        self.fc1_dims = 128
        self.fc2_dims = 256
        self.fc3_dims = 512
        self.fc4_dims = 256
        self.fc5_dims = 128

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.q = nn.Linear(self.fc5_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta) # self.parameter() come from nn.Module
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc3(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc4(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc5(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_chkpt_file))

class ActorNetwork(nn.Module): # state only
    def __init__(self, alpha, input_dims, max_action=1, n_actions=2, name='actor', load_chkpt_dir='tmp/td3_load', save_chkpt_dir='tmp/td3_save'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.load_chkpt_dir = load_chkpt_dir
        self.load_chkpt_file = os.path.join(self.load_chkpt_dir, name+'_td3')
        self.save_chkpt_dir = save_chkpt_dir
        self.save_chkpt_file = os.path.join(self.save_chkpt_dir, name+'_td3')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1_dims = 128
        self.fc2_dims = 256
        self.fc3_dims = 512
        self.fc4_dims = 256
        self.fc5_dims = 128


        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.mu = nn.Linear(self.fc5_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = self.fc3(prob)
        prob = F.relu(prob)
        prob = self.fc4(prob)
        prob = F.relu(prob)
        prob = self.fc5(prob)
        prob = F.relu(prob)
        mu = T.tanh(self.mu(prob)) 

        return mu

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_chkpt_file))