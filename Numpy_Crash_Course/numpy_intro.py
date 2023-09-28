#! usr/bin/python3
import numpy as np

def main():
    add_two_vectors_using_lists()
    add_two_vectors_using_numpy()
    
def add_two_vectors_using_lists():
    list_two = list(range(1,4))
    list_three = list(range(1,4))
    list_sum = []

    for i in range(3):
        list_two[i] = list_two[i]**2
        list_three[i] = list_three[i]**3
        list_sum.append(list_two[i] + list_three[i])
        
    print(list_sum)

def add_two_vectors_using_numpy():
    array_two = np.arange(1,4) ** 2
    array_three = np.arange(1,4) ** 3
    array_sum = array_two + array_three
    print(array_sum)
    sample_array = np.array([1,2,3,4,5])
    print(np.power(sample_array, 4))
    print(np.negative(sample_array))
    print(np.exp(sample_array))
    print(np.log(sample_array))
    print(np.sin(sample_array))
    
    
main()