# Parallel Programming

### Day 1 - 24/06/2024

1. OpenMP/Lab1-hello.c   - OpenMP program to print Thread ID and no. of threads executing in parallel
2. OpenMP/Lab2-sum.c   - OpenMP program to parallelize calculating sum over a for loop using "reduction(+:sum)"
3. OpenMP/Lab3-stream/* - OpenMP program to find memory bandwidth benchmark 

-------------------------------------------------------------
##### export OMP_NUM_THREADS=1
-------------------------------------------------------------

Function  Best Rate MB/s  Avg time Min time Max time

Copy: 10015.3 0.198250 0.159756 0.252664

Scale:  10121.3 0.388264 0.158083 1.809805

Add: 9986.3 0.284094 0.240329 0.348430

Triad:  10465.5 0.259827 0.229325 0.296106

-------------------------------------------------------------
##### export OMP_NUM_THREADS=8
-------------------------------------------------------------

Function  Best Rate MB/s  Avg time Min time Max time

Copy: 16243.2 0.141014 0.098503 0.193952

Scale:  21224.7 0.131214 0.075384 0.225820

Add:  24654.1 0.168321 0.097347 0.277322

Triad:  26564.8 0.139407 0.090345 0.202934

-------------------------------------------------------------

### Day 2 - 25/06/2024

