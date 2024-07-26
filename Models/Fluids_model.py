import torch
from torch import nn
import torch.nn.functional as F

# class MLP(nn.Module):
    
#     def __init__(self, dim_feat, drop_rate=0.56):
#         super(MLP, self).__init__()

        
#         # label predictor
#         self.class_classifier = nn.Sequential()
#         self.class_classifier.add_module('c_fc1', nn.Linear(dim_feat, 32))
#         self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(32))
#         self.class_classifier.add_module('c_relu1', nn.ReLU(True))
#         self.class_classifier.add_module('c_drop1', nn.Dropout(drop_rate))
#         self.class_classifier.add_module('c_fc2', nn.Linear(32, 16))
#         self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(16))
#         self.class_classifier.add_module('c_relu2', nn.ReLU(True))
#         self.class_classifier.add_module('c_fc3', nn.Linear(16, 4))

#     def forward(self, feature):
        
#         class_output = self.class_classifier(feature)
    
#         return class_output
    
    
class MLP(nn.Module):
    def __init__(self, input_size, drop_rate, hidden_unit_sizes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_unit_sizes = hidden_unit_sizes
        
        leaky_relu = nn.LeakyReLU()
        
        layer = [nn.Linear(input_size, hidden_unit_sizes[0]),
                   nn.BatchNorm1d(hidden_unit_sizes[0]),
                   nn.Dropout(drop_rate),
                   leaky_relu]
        
        for i in range(1,len(hidden_unit_sizes)):
        
                layer.append(nn.Linear(hidden_unit_sizes[i-1], hidden_unit_sizes[i]))
                layer.append(nn.BatchNorm1d(hidden_unit_sizes[i]))
                layer.append(nn.Dropout(drop_rate))
                layer.append(leaky_relu)
                
        self.layer = nn.Sequential(
            *layer)
        
        
    def forward(self, x):
        output = self.layer(x)
        return output