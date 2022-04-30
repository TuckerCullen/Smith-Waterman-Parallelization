import numpy as np 
import pandas as pd 
from random import choice
import time 
from mpi4py import MPI

GAP_PENALTY = -2

# saw mixed messages for what these two values should be, went off wikipedias values: 
MISMATCH_PENALTY = -3
MATCH_REWARD = 3

def print_matrix(alignment_matrix, query, reference):
    """ helper function for printing matrix with nuclotide labels"""

    ind = list(query)
    col = list(reference)

    col.insert(0, " ")
    ind.insert(0, " ")

    alignment_df = pd.DataFrame(alignment_matrix, columns=col, index=ind, dtype="int64")

    print(alignment_df)

def rand_DNA(desired_length):
    """ helper for generating random dna sequences """

    DNA=""
    for i in range(desired_length):

        DNA += choice("CGTA")
    
    return DNA


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



####################################################################
def fill_alignment_matrix_mpi(query, reference):

    M = len(query)
    N = len(reference)

    # alignment_matrix = np.zeros(shape=(M+1, N+1))

    if rank != 0 and rank <= N:

        print("Call diagnonal_wavfront on rank ", rank)

        col = []
        diagonal_wavefront(col, query, reference)


    if rank == 0:

        alignment_lists = []
        for p in range(1, N+1):

            col = comm.recv(source=p, tag=p)

            alignment_lists.append(col)

            alignment_matrix = np.array(alignment_lists).T

            alignment_matrix = np.pad(alignment_matrix, ((1, 0), (1, 0)), 'constant')

        return alignment_matrix


def diagonal_wavefront(col : list, query, reference, diag=0, up=0, step=0):

    M = len(query)
    N = len(reference)

    if step == M:

        comm.send(col, dest=0, tag=rank)

        # # if there are columns without processors, take over the next column
        # if 
        #     diagonal_wavefront(col, query, reference)

        return 

    print(f"iteration number {len(col)} on rank {rank}")

    if rank == 1:
        left = 0
    else:
        left = comm.recv(source=rank-1, tag=step)

    if query[step] == reference[rank - 1]:
        diag_adjustment = MATCH_REWARD
    else:
        diag_adjustment = MISMATCH_PENALTY

    diag_adjust = max(0, diag + diag_adjustment)
    up_adjust = max(0, up + GAP_PENALTY)
    left_adjust = max(0, left + GAP_PENALTY)
    
    
    alignment_score = max(diag_adjust, up_adjust, left_adjust )
    col.append(alignment_score)

    if rank < size - 1:
        comm.send(alignment_score, dest=rank+1, tag=step)

    step += 1

    diagonal_wavefront(col, query, reference, diag=left, up=alignment_score, step=step)

    ##################################################################################

def smith_waterman(query, reference, verbose=True): 
    """
    main function 
    """

    if rank == 0:
        total_start = time.perf_counter()
        matrix_start = time.perf_counter()

    alignment_matrix = fill_alignment_matrix_mpi(query, reference) # compute alignment matrix 

    
    if rank == 0:

        print(alignment_matrix)

        print_matrix(alignment_matrix, query, reference)

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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    # print("Rank: ", rank)
    # print("Size: ", size)

    # smith_waterman(query="ATCG", reference="GCTA")

    #test case from bioinformatica videos 
    # smith_waterman(query="AGCT", reference="ATGCT")
    
    # smith_waterman(query="AGCTAT", reference="ATGC")

    # test case from wikipedia: 
    smith_waterman(query="GGTTGACTA", reference="TGTTACGG")

    # print("Scaling Tests: ------------------------------------------ ")

    #testing on random DNA sequences 
    # smith_waterman( rand_DNA(10), rand_DNA(10), verbose= False)
    # smith_waterman( rand_DNA(100), rand_DNA(100), verbose= False)
    # smith_waterman( rand_DNA(1000), rand_DNA(1000), verbose= False)
    # smith_waterman( rand_DNA(10000), rand_DNA(10000), verbose= False) # this takes about 6 min











