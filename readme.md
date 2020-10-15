# COPML

COPML provides a framework of a collaborative machine learning where multiple data-owners wish to jointly train a logistic regression model, while keeping their individual datasets private from the other parties.

For details of our work, please see our paper "A Scalable Approach for Privacy-Preserving Collaborative Machine Learning," accepted to NeurIPS 2020.



### Usage

There are six main codes to run the experiments on the Amazon EC2 cloud. 
Therefore, before using the code, an Amazon EC2 cluster needs to be created first with N+1 machine instances. 
This can be done by using the starcluster tool, http://star.mit.edu/cluster/.
In the experiments of the paper, m3.xlarge machine instances were used. 

MPI4Py package should be installed at each machine 



The detailed usage of the six main codes are as follows:


1. Masterless_COPML_CIFAR.py
    - Goal        : get the raw data of training time of COPML with 50 iterations
    - Command : mpiexec -n (N+1) python2 Masterless_COPML_CIFAR.py (N)
    - input       : ./datasets/cifar-10-batches-py/*
    - output     : CPML_CIFAR_(N)

2. Masterless_BH_CIFAR.py
    - Goal        : get the raw data of training time of MPC based on BH with 50 iterations
    - Command : mpiexec -n (N+1) python2 Masterless_BH_CIFAR.py (N)
    - input       : ./datasets/cifar-10-batches-py/*
    - output     : BH_CIFAR_(N)

3. Masterless_BGW_CIFAR.py
    - Goal        : get the raw data of training time of MPC based on BGW with 50 iterations
    - Command : mpiexec -n (N+1) python2 Masterless_BGW_CIFAR.py (N)
    - input       : ./datasets/cifar-10-batches-py/*
    - output     : BGW_CIFAR_(N)

4. Masterless_COPML_GISETTE.py
    - Goal        : get the raw data of training time of COPML with 50 iterations
    - Command : mpiexec -n (N+1) python2 Masterless_CodedPrivateML_GISETTE.py (N)
    - input       : ./datasets/Gisette/*
    - output     : CPML_GISETTE_(N)

5. Masterless_BH_GISETTE.py
    - Goal        : get the raw data of training time of MPC based on BH with 50 iterations
    - Command : mpiexec -n (N+1) python2 Masterless_BH_GISETTE.py (N)
    - input       : ./datasets/Gisette/*
    - output     : BH_GISETTE_(N)

6. Masterless_BGW_GISETTE.py
    - Goal        : get the raw data of training time of MPC based on BGW with 50 iterations
    - Command : mpiexec -n (N+1) python2 Masterless_BGW_GISETTE.py (N)
    - input       : ./datasets/Gisette/*
    - output     : BGW_GISETTE_(N)



- Download CIFAR dataset from https://www.cs.toronto.edu/~kriz/cifar.html and unzip it into ./datasets/cifar-10-batches-py/
- Download GISETTE dataset from https://archive.ics.uci.edu/ml/datasets/Gisette and unzip it into ./datasets/Gisette/



### Citation

Will be updated.



### Contact

The corresponding author is: 

Jinhyun So 

jinhyuns@usc.edu or jinhyun.soh@gmail.com