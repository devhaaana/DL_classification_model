import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from setting_list import *

        
class Set_dataset():
    def __init__(self, args):
        super(Set_dataset).__init__()
        self.args = args
        self.input_dim = 0
        self.target_dim = 0
    
    def get_input_dimension(self):
        if self.args.dataset == data_list[0] or self.args.dataset == data_list[1]:
            self.input_dim = (32, 32)
        elif self.args.dataset ==data_list[2]:
            self.input_dim = (224, 224)
        elif self.args.dataset ==data_list[3] or self.args.dataset ==data_list[4]:
            self.input_dim = (28, 28)
        elif self.args.dataset ==data_list[5]:
            self.input_dim = (32, 32)
        elif self.args.dataset ==data_list[6]:
            self.input_dim = (96, 96)
        
        return self.input_dim

    def get_target_dimension(self):
        if self.args.dataset == data_list[0]:
            self.target_dim = 10
        elif self.args.dataset ==data_list[1]:
            self.target_dim = 100
        elif self.args.dataset ==data_list[2]:
            self.target_dim = 1000
        elif self.args.dataset ==data_list[3]:
            self.target_dim = 10
        elif self.args.dataset ==data_list[4]:
            self.target_dim = 10
        elif self.args.dataset ==data_list[5]:
            self.target_dim = 10
        elif self.args.dataset ==data_list[6]:
            self.target_dim = 10
            
        return self.target_dim