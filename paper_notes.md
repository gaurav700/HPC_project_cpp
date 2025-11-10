# Paper: PERFORMANCE EVALUATION OF PARALLEL MESSAGE PASSING AND THREAD PROGRAMMING MODEL ON MULTICORE ARCHITECTURES
- ## Problem:  
    Previous version of MPI which is MPI-1 has no shared memory concept, and Current MPI version 2 which is MPI-2 has a limited support for shared memory systems. In this research, MPI-2 version of MPI will be compared with OpenMP to see how well does MPI perform on multicore / SMP (Symmetric Multiprocessor) machines.
    - 1. How does different parallel programming model influenced parallel performance on different memory architecture?
    - 2. How does workshare construct differ between shared and distributed shared memory systems?   
- ## Hardware: 
    - For OpenMP and sequential program, test was done on a single Intel core duo 1.7 Ghz T5300 laptop with 1GB RAM running on linux SUSE. 
    - For MPI program test was done on a two Intel core duo laptops one with frequency 1.7 GHz T5300 laptop with 1GB RAM running on Windows Xp SP3 and another one with frequency 1.83 GHz T5500 with 1GB RAM running on Windows Xp SP3
- ## Method: 
    For OpenMP, a singlemulticore machines with two worker cores will be used to calculated the matrix. For the MPI, two multicore machines with three worker cores will be used (one asa master process who decompose the matrix to sub - matrix and distrbute it to two other worker process and compose the final result matrix from the sub - matrix multiplication done by its two worker process).

- ## Matrix sizes: 
    Application used as a testing is N X N rectangular matrix multiplication with adjustable matrix dimension N ranging from 10 to 2000. 
- ## Key results (figures / numbers):
    - ### Running time comparison with various matrix dimension
        | Matrix Dimension (N) | Sequential (s) | OpenMP (s) | MPI (s) |
        |----------------------|----------------|------------|---------|
        |100                   |  0.03          | 0.02       | 0.33    |
        |500                   |  2.09          | 1.11       | 1.52    |
        |1000                  |  22.52         | 14.36      | 8.19    |
        |2000                  |  240.97        | 163.6      | 60.19   |

    - ### Throughput comparison with various matrix dimension
        | Matrix Dimension (N) | Sequential (MFLOPS) | OpenMP (MFLOPS) | MPI (MFLOPS) |
        |----------------------|---------------------|-----------------|--------------|
        |100                   |  59.08              | 86.82           | 6.07         |
        |500                   |  119.54             | 224.35          | 164.85       |
        |1000                  |  88.77              | 139.17          | 244.17       |
        |2000                  |  8.91               | 13.17           | 265.77       |

    - ### Speed comparison with various matrix dimension
        | Matrix Dimension (N) | OpenMP (s) | MPI (s) | 
        |----------------------|------------|---------|
        |100                   |  1.47      | 0.1     |
        |500                   |  1.88      | 1.38    |
        |1000                  |  1.57      | 2.75    |     
        |2000                  |  1.47      | 4       |
- ## Notes / what to emulate or avoid: 
    To avoid memory inconsistency, there is a cache memory which can be shared to all processor. Shared cache memory canbe used for each core processor to write and read data.
