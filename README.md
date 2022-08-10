# Parallizing the Smith-Waterman Sequence Alignment Algorithm Using `mpi4py`

The Smith-Waterman (SW) local sequence alignment algorithm is one of the central algorithms in bioinformatics. While the algorithm is guaranteed to find the best local alignment between two sequences, it suffers from an extremely slow quadratic runtime. As such, less-sensitive heuristic algorithms are often used in its place. There remains a need for faster, parallel implementations of Smith-Waterman to make it a feasible option for larger sequence alignment tasks.

This repository contains three implementations of the SW algorithm: a serial implementation (`serial-sw.py`), and two parallel implementations (`blocking-mpi-sw.py` and `diag-mpi-sw.py`). The parallel implementations utilize the message passing interface (MPI) protocol for distributed memory computing. They were developed in python using the `mpi4py` library. 

## Running each implementation:

### Serial 

To run on a custom sequence:

`python serial-sw.py -query <query sequence> -reference <reference sequence>`

To run on the main example sequences from wikipedia: 

`python serial-sw.py --test 1`

To run on two randomly generated sequences of a given length (for scaling testing):

`python serial-sw.py --length <int of length>`

### Blocking 

### Diagonal Wavefront 