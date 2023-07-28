# Created by: nbujiashvili

# retrieving the additional factors in case the user wants to scale different variables in the simulation


def get_time_factor():
    coef = input("Enter Factor For Simulation Time (1- Normal, 1.5- Extended, 2- Super Extended): ")
    numC = float(coef)
    coef_2 = input("Enter Factor For Sigma (1- Normal, >1 - Shorter time for Spontaneous Neurons, <1- Extended Time): ")
    numC_2 = float(coef_2)
    graph_type = input("Enter a preferred graph type (0 - circular, 1- random, 2- shell, 3- spectral, other- spring/default): ")
    numC_3 = int(graph_type)
    stoch_factor = input("Enter a factor for Stochasticity constant: ")
    numC_4 = float(stoch_factor)
    return numC, numC_2, numC_3, numC_4

# Retrieving the name for the csv file that will be used for the adjacency matrix 
def csv_file_entry():
    file_name = input("CSV file name for Adjacency Matrix (without spaces): ")
    return file_name