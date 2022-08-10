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
    """ Given a complete alignment matrix, 

    Args:
        alignment_matrix (np.array): a filled-in alignment matrix (i.e. the output of fill_alignment_matrix)
        query (str): the query sequence 
        reference (str): the reference sequence
        recursive (bool, optional): _description_. Defaults to False.

    Returns:
        (str, str): local alignments of the two strings, 
    """

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


def fill_alignment_matrix_mpi(query, reference):

    M = len(query)
    N = len(reference)

    # alignment_matrix = np.zeros(shape=(M+1, N+1))

    # calculate how many times each processor needs to run 
    iters = N // (size - 1)
    leftovers = N % (size - 1)

    if rank != 0 and rank <= N:

        # some processors need to run one more time to finish up the last few columns 
        if rank <= leftovers:
            iters_left = iters + 1
        # all the processors will need to run on "iters" number of columns. 
        else:
            iters_left = iters
            
        col_index = rank - 1
        while iters_left > 0:
            col = []
            iters_left -= 1

            if col_index == rank - 1:
                diagonal_wavefront(col, query, reference, col_index, first_pass=True) 
            else: 
                diagonal_wavefront(col, query, reference, col_index, first_pass=False) 
            col_index = col_index + (size-1) 

    if rank == 0:

        alignment_lists = []

        while iters > 0:

            for p in range(1, size):
                col = comm.recv(source=p, tag=p)
                alignment_lists.append(col)
            iters -= 1

        # recieve the last few columns 
        for p in range(1, leftovers + 1):
            col = comm.recv(source=p, tag=p)
            alignment_lists.append(col)


        alignment_matrix = np.array(alignment_lists).T

        alignment_matrix = np.pad(alignment_matrix, ((1, 0), (1, 0)), 'constant')

        #print("reached here")

        return alignment_matrix


def diagonal_wavefront(col : list, query : str, reference : str, cur_col_index, first_pass=True):

    """
    Arguments:
        col : list - the list of alignment scores for the current column being calculated
        query : str - the query sequence
        reference : str - the reference sequence 
        cur_col_index : int - the index of the current column (for indexing the reference sequence)
        first_pass : bool - says if this is the first column the current process is calculating. Used for telling rank 1 if it should initalize left as 0 or recieve a value
    """

    M = len(query)
    N = len(reference)

    step = 0
    left = 0
    up = 0
    diag = 0

    while step <= M - 1: 
        if rank == 1:
            if first_pass:
                left = 0
            else:
                left = comm.recv(source=size-1, tag=step)
                # print(f"Rank 1 recieving left from rank {size-1}. left={left}, tag={step}")
        else:
            left = comm.recv(source=rank-1, tag=step)
            # print(f"Rank {rank} recieving left from rank {rank-1}. left={left}, tag={step}")

        if query[step] == reference[cur_col_index]:
            diag_adjustment = MATCH_REWARD
        else:
            diag_adjustment = MISMATCH_PENALTY

        diag_adjust = max(0, diag + diag_adjustment)
        up_adjust = max(0, up + GAP_PENALTY)
        left_adjust = max(0, left + GAP_PENALTY)
        
        alignment_score = max(diag_adjust, up_adjust, left_adjust)
        col.append(alignment_score)

        diag = left 
        up = alignment_score

        # outer if statement accounts for the case where the number of processors is greater than the reference sequence length. 
        if rank <= size - 1:

            # if the current rank is the last one, send "left" back around to rank 1 
            if rank == size-1: 
                send_to = 1
            else:
                send_to = rank+1

            comm.send(alignment_score, dest=send_to, tag=step)
        #print("end")
        step += 1
    
    if step == M:
        comm.send(col, dest=0, tag=rank)
        #print(f"sending column {cur_col_index} to rank 0 from rank {rank}", col)
        return 


    #########################################################################################################################

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
    parser.add_argument("--test", "-t", type=int, default=0)

    args = parser.parse_args()

    if args.test:
        smith_waterman(query="GGTTGACTA", reference="TGTTACGG", verbose=True)

    else: 
        seq1 = rand_DNA(args.length, seed = args.seed)
        seq2 = rand_DNA(args.length, seed = args.seed)
        smith_waterman(seq1, seq2, False)