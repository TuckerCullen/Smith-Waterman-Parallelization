import numpy as np
import argparse

# TO RUN, use python serial_SmithWaterman.py -s1 "ACTG" -s2 "ACTGATTCG" 
# s1, s2 are the sequnce strings

def smith_waterman_alg(s1, s2):
    num_rows = len(s1)
    num_cols = len(s2)
    val_matrix = np.zeros(shape=(num_rows + 1, num_cols + 1), dtype= int)
    trace_matrix = np.empty(shape = (num_rows + 1, num_cols + 1), dtype = np.dtype('U100'))
    trace_matrix[0][0] = "STOP"
    # -1 works because in this algorithm, lowest val is 0
    max_val = -1
    index = (-1,-1)

    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            if(s1[i-1] == s2[j-1]):
                temp = 1
            else: 
                temp = -1
            
            diag_val = val_matrix[i-1, j-1] + temp
            top_val = val_matrix[i-1, j] + -1
            left_val = val_matrix[i, j-1] + -1
            val_matrix[i,j] = max(diag_val, top_val, left_val, 0)

            if val_matrix[i,j] == 0:
                trace_matrix[i,j] = "STOP"
            elif val_matrix[i,j] == left_val:
                trace_matrix[i,j] = "LEFT"
            elif val_matrix[i,j] == top_val:
                trace_matrix[i,j] = "TOP"
            elif val_matrix[i,j] == diag_val:
                trace_matrix[i,j] = "DIAG"
            
            if(val_matrix[i,j] > max_val):
                max_val = val_matrix[i,j]
                index = (i,j)

    aligned_s1 = ""
    aligned_s2 = ""
    i = index[0]
    j = index[1]
    temp1 = 0 
    temp2 = 0
    print(trace_matrix)
    print(i)
    print(j)

    while trace_matrix[i, j] != "STOP":
        if trace_matrix[i,j] == "DIAG":
            temp1 = s1[i-1]
            temp2 = s2[j-1]
            i -= 1
            j -= 1
        elif trace_matrix[i,j] == "TOP":
            temp1 = s1[i-1]
            temp2 = "-"
            i -= 1
        elif trace_matrix[i,j] == "LEFT":
            temp1 = "-"
            temp2 = s2[j-1]
            j -= 1
        
        aligned_s1 += temp1
        aligned_s2 += temp2

    aligned_s1=''.join(reversed(aligned_s1))
    aligned_s2=''.join(reversed(aligned_s2))

    return (aligned_s1, aligned_s2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A test program.')

    parser.add_argument("-s1", "--sequence_1", default="ACTG")
    parser.add_argument("-s2", "--sequence_2", default="ACTG")

    args = parser.parse_args()
    
    (aligned_s1, aligned_s2) = smith_waterman_alg(args.sequence_1 , args.sequence_2)
    
    print(aligned_s1)
    print(aligned_s2)
        




    



             
    

