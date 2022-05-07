import numpy as np 
import pandas as pd 
import random
import time 
import argparse
from mpi4py import MPI

GAP_PENALTY = -2

# saw mixed messages for what these two values should be, went off wikipedias values: 
MISMATCH_PENALTY = -3
MATCH_REWARD = 3

### MPI INITIALIZATION ###
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def print_matrix(alignment_matrix, query, reference):
    """ helper function for printing matrix with nuclotide labels"""

    ind = list(query)
    col = list(reference)

    col.insert(0, " ")
    ind.insert(0, " ")

    alignment_df = pd.DataFrame(alignment_matrix, columns=col, index=ind, dtype="int64")

    print(alignment_df)

def rand_DNA(desired_length, chars = 'CGTA', seed = 0):
    """ helper for generating random dna sequences """  
    random.seed(seed)
    return ''.join(random.choice(chars) for _ in range(desired_length))


def fill_alignment_matrix(query, reference):
    """
    Creates the alignment matrix. 
    """

    M = len(query)
    N = len(reference)

    alignment_matrix = np.zeros(shape=(M+1, N+1))

    for i in range(1, M+1):
        for j in range(1, N+1):

            print(i, j)

            alignment_score, movement = next_movement(alignment_matrix, i, j, query, reference)

            alignment_matrix[i][j] = alignment_score

    return alignment_matrix


def fill_alignment_matrix_diag(query, reference):
    """
    Fills the alignment matrix one diagonal at a time. 
    """

    M = len(query)
    N = len(reference)

    alignment_matrix = np.zeros(shape=(M+1, N+1))

    for diag in range(1, (M+N)):

        start_col = max(0, diag - M)

        diag_len = min(diag, (N - start_col), M)

        for k in range(0, diag_len):

            i = min(M, diag) - k
            j = start_col + k + 1

            alignment_score, movement = next_movement(alignment_matrix, i, j, query, reference)

            alignment_matrix[i][j] = alignment_score

            # print( i , j, end='\t')
        
        # print("\n")
            
    return alignment_matrix




def next_movement(alignment_matrix, i, j, query, reference):
    """
    - given a position in the alignment_matrix, determines what the score should be and from where that score came from
    - returns:
        alignment_score - int, the score for the current position
        movement - int, records what direction the alignment_score came from. 0 corresponds to left, 1 to diagonal, and 2 to up 

    """

    # check if the nucleotide at the current position matches between query and reference 
    if query[i-1] == reference[j-1]:
        diag_adjustment = MATCH_REWARD
    else:
        diag_adjustment = MISMATCH_PENALTY

    # assign gap 
    left = max(0, alignment_matrix[i][j-1] + GAP_PENALTY)
    diag = max(0, alignment_matrix[i-1][j-1] + diag_adjustment)
    up = max(0, alignment_matrix[i-1][j] + GAP_PENALTY) 

    movement_list = [left, diag, up]

    alignment_score = max(movement_list)

    movement = movement_list.index(alignment_score)

    return alignment_score, movement


def traceback(alignment_matrix, query, reference, recursive=False):

    aligned_query = []
    aligned_ref = []

    # find the location of the largest score in the matrix (start from here)
    largest_score_pos = np.unravel_index(alignment_matrix.argmax(), alignment_matrix.shape)

    # for large sequences, the max python recursive depth is reached so an iterative approach is prefered
    if recursive:
        traceback_recursive(alignment_matrix, largest_score_pos, aligned_query, aligned_ref, query, reference)

    else:
        # iteratively trace the alignment matrix back, appending the proper nucleotides to aligned_query and aligned_ref
        traceback_iterative(alignment_matrix, largest_score_pos, aligned_query, aligned_ref, query, reference)

    # print the final local sequence alignment
    return aligned_query, aligned_ref


def traceback_recursion(alignment_matrix, position, aligned_query, aligned_ref, query, reference):
    """
    recursive implemention of the traceback function 
    """

    # Uncomment to debug route that traceback travels:
    # print(position)

    i, j = position

    # base case, stop when you reach a 0 in the traceback
    if alignment_matrix[i][j] == 0:
        return

    # determine where the current score came from, and trace it back that same way
    alignment_score, movement = next_movement(alignment_matrix, i, j, query, reference)

    # trace back to the left
    if movement == 0:
        aligned_query.append(query[i-1])
        aligned_ref.append("-")
        new_position = (i, j-1)

    # traceback diagonally
    if movement == 1:
        aligned_query.append(query[i-1])
        aligned_ref.append(query[i-1])
        new_position = (i-1, j-1)

    #traceback up 
    if movement == 2:
        aligned_query.append(query[i-1])
        aligned_ref.append("-")
        new_position = (i-1, j)

    # recurse on new position 
    traceback_recursion(alignment_matrix, new_position, aligned_query, aligned_ref, query, reference)

def traceback_iterative(alignment_matrix, position, aligned_query, aligned_ref, query, reference):
    """
    iteratively traceback through the matrix 
    """

    # Uncomment to debug route that traceback travels:
    # print(position)

    # you can't have a local alignment longer than the shortest of the two sequences 
    max_alignment_length = min(len(query), len(reference))

    new_position = position

    for step in range(max_alignment_length):

        i, j = new_position

        # base case, stop when you reach a 0 in the traceback
        if alignment_matrix[i][j] == 0:
            return

        # determine where the current score came from, and trace it back that same way
        alignment_score, movement = next_movement(alignment_matrix, i, j, query, reference)

        # trace back to the left
        if movement == 0:
            aligned_query.append(query[i-1])
            aligned_ref.append("-")
            new_position = (i, j-1)

        # traceback diagonally
        if movement == 1:
            aligned_query.append(query[i-1])
            aligned_ref.append(query[i-1])
            new_position = (i-1, j-1)

        #traceback up 
        if movement == 2:
            aligned_query.append(query[i-1])
            aligned_ref.append("-")
            new_position = (i-1, j)



###################################################################################
########################### PARALLEL FUNCITONS BELOW ##############################

# 10 x 10 
# 4 procs 
# cols_per_proc = 2 leftovers = 2

# rank * cols_per_proc + (leftovers - rank)

# 1 - 3 : ref[0:3]  rank * cols_pert_proc = 3
# 2 - 3 : ref[3:6]  rank * cols_pert_proc = 6
# 3 - 2 : ref[6:8]  rank * cols_pert_proc = 6 -> 8
# 4 - 2 : ref[8:10] rank * cols_pert_proc = 8 -> 10

def fill_alignment_matrix_mpi(query, reference):

    M = len(query)
    N = len(reference)

    cols_per_proc = N // (size - 1)
    leftovers = N % (size - 1)


    if rank != 0 and rank <= N:

        if rank <= leftovers:
            cols_per_proc += 1
            end_ref_i = rank * cols_per_proc
        else:
            end_ref_i = (rank * cols_per_proc) + leftovers

        start_ref_i = end_ref_i - cols_per_proc

        # print(f"rank {rank} operating on subreference: {(start_ref_i, end_ref_i)}")

        blocking(cols_per_proc, query=query, sub_reference=reference[ start_ref_i : end_ref_i ])

        return


    # recieve sub-matrices from each rank and concatenate 
    if rank == 0:

        # the array recieved from rank 1 has an extra column (of zeros)
        if 1 <= leftovers:
            one_size = cols_per_proc + 2
        else:
            one_size = cols_per_proc + 1 

        # must initialize empty buffer before recieving numpy array (M+1 accounts for first row of 0s)
        alignment_matrix = np.empty((M + 1, one_size), dtype='i')

        comm.Recv([alignment_matrix, MPI.INT], source=1, tag=1)
        
        # to account if there are more processors than columns 
        num_workers = min(size, N)

        for worker_rank in range(2, num_workers):

            if worker_rank <= leftovers:
                worker_size = cols_per_proc + 1
            else:
                worker_size = cols_per_proc

            # initialize buffer to be filled by Recv
            recieved_submatrix = np.empty((M + 1, worker_size), dtype='i')

            comm.Recv([recieved_submatrix, MPI.INT], source=worker_rank, tag=worker_rank)

            # print(f"Recieved submatrix from rank {worker_rank}: ")

            alignment_matrix = np.concatenate((alignment_matrix, recieved_submatrix), axis=1)

        return alignment_matrix


def blocking(cols_per_proc, query, sub_reference):

    M = len(query)
    N = len(sub_reference)

    sub_matrix = np.zeros(shape=(M+1, cols_per_proc+1), dtype='i')

    for i in range(1, M+1):
        if rank != 1:
            sub_matrix[i][0] = comm.recv(source=rank-1, tag=i)
        for j in range(1, cols_per_proc+1):

            alignment_score, _ = next_movement(sub_matrix, i, j, query, sub_reference)
 
            sub_matrix[i][j] = alignment_score

        if rank < (size - 1):
            comm.send(alignment_score, dest=rank+1, tag=i)

    if rank != 1:
        sub_matrix = sub_matrix[:, 1:]
        sub_matrix = np.ascontiguousarray(sub_matrix, dtype='i')

    # print()
    # print(f"Sub - matrix completed by rank {rank}: ")
    # print(sub_matrix)
    # print()
    
    comm.Send([sub_matrix, MPI.INT], dest=0, tag=rank)

    # print(f"Submatrix sent by rank {rank}")
    return 


def smith_waterman(query, reference, verbose=True): 
    """
    main function 
    """

    if rank == 0:
        total_start = time.perf_counter()
        matrix_start = time.perf_counter()

    alignment_matrix = fill_alignment_matrix_mpi(query, reference) # compute alignment matrix 

    if rank == 0:

        matrix_end = time.perf_counter()
        traceback_start = time.perf_counter()

        aligned_query, aligned_ref = traceback(alignment_matrix, query=query, reference=reference) # traceback 

        traceback_end = time.perf_counter()
        total_end = time.perf_counter()

        if verbose:
            print()
            print("Query: ", query)
            print("Reference: ", reference, "\n")
            print_matrix(alignment_matrix, query, reference)
            print()
            print(aligned_query)
            print(aligned_ref)
        
        print()
        print("Problem Size: ", len(query), " x ", len(reference))
        print("Time to create fill alignment matrix -: ", matrix_end - matrix_start)
        print("Time for traceback -------------------: ", traceback_end - traceback_start)
        print("Total runtime ------------------------: ", total_end - total_start)
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--length", "-l", type=int, default=10)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--check", "-c", type=int, default=0)

    args = parser.parse_args()

    if args.check:
        smith_waterman(query="GGTTGACTA", reference="TGTTACGG", verbose=True)
    else: 
        seq1 = rand_DNA(args.length, seed = args.seed)
        seq2 = rand_DNA(args.length, seed = args.seed)
        smith_waterman(seq1, seq2, False)