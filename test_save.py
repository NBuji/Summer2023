# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# import networkx as nx
# from networkx.algorithms import community
# from networkx.algorithms.community.centrality import girvan_newman
# from networkx.algorithms.community.quality import modularity
# import metis
# import sys
# import os
# import time
# import re
# from si_prefix import si_format
# from snn_constants_equations import *
# from snn_functions import *
# from example_graphs import *
# from snn_functions import *
# from gpp_snn import *
# from partition_functions import *
# from time_factor_input import *




# if len(sys.argv) < 3:
#     print("Insufficient parameters")
#     sys.exit()

# graph_type = sys.argv[1] # graph type (trivial, WS, WS_SF, ER, SF)
# N = int(sys.argv[2]) # number of nodes
# from datetime import datetime
# dt_now = datetime.now() # datetime object containing current date and time

# # check for valid graph type
# if graph_type not in graph_types:
#     print("Invalid graph type has been selected")
#     sys.exit()

# graph_name = f'{N}_{graph_type}'
# # append graph name if already exists
# my_parent_path = f'{save_to}{graph_name}'
# while os.path.exists(my_parent_path):
#     match = re.search(r'\d+$', my_parent_path)
#     if match:
#         # Get the numeric portion of the string
#         num_str = match.group()
#         # Increment the numeric portion of the string
#         num = int(num_str) + 1
#         # Get the non-numeric portion of the string
#         prefix = my_parent_path[:match.start()]
#         # Concatenate the non-numeric and incremented numeric portion of the string
#         my_parent_path =  f"{prefix}{num:0{len(num_str)}d}"
#     else:
#         # If the string does not end with a numeric portion, simply append "1"
#         my_parent_path =  f"{my_parent_path}_1"
# os.makedirs(my_parent_path)

# # optional arguments

# with open(f'{my_parent_path}/information123.txt', 'a') as f:
#     f.write("the number of nodes is less than 4. This graph is meaningless \n")
#     f.close()
# sys.exit()
# integ1 = 14
# coef = 4
# coef = int(coef)
# integ1 = integ1*coef
# print(integ1)
# print(coef)
# m  = get_time_factor()
# print("Time Factor: ")
# print(m)


from inspect import GEN_SUSPENDED
import numpy as np
import math
# from jj_plot_functions import *
from brian2 import *
import random
from snn_functions import *
# M_1 = get_spike_matrix()
# print(M_1)
# print(M_1[0][19])
k1 = get_matrix_bias_current(0.37,11)
print(k1)
# N_0 = 10
# N_1 = 10
# Heights = np.linspace(0, 1, N_0)
# bias_current = 1.9

# bias_current_vals = np.linspace(1.6, 1.9, N_1 )
# A = np.zeros((len(Heights), len(bias_current_vals)), dtype=float)
# B = np.zeros((len(Heights), len(bias_current_vals)), dtype=float)
# for i, height_1 in enumerate(Heights):
#     for j, bias_cur_1 in enumerate(bias_current_vals):
#         A[i][j] = height_1
#         B[i][j] = bias_cur_1
        

# # folder_path = './side_tests/'
# # file_name_1 = folder_path + 'matrix.csv'
# # np.savetxt(file_name_1, A, delimiter=',')
# # file_name_2 = folder_path + 'matrix_2.csv'
# # np.savetxt(file_name_2, B, delimiter=',')
# print(A[0][8])
# print(A[8][0])
# bias_current_marks = np.linspace(1.6, 1.9, 20 )
# print(bias_current_marks)

