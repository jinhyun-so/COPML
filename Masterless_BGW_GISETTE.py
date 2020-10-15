from mpi4py import MPI
import numpy as np
import random
from array import array
import math
import time
import sys
import gc
import os

import matplotlib as mpl
import matplotlib.pylab as plt
import pickle as pickle

from utils.mpc_function import *
from utils.polyapprox_function import *



# system parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) == 1:
    if rank ==0:
        print("ERROR: please input the number of workers")
    exit()
else:
    N = int(sys.argv[1])

K = 1
T = int(np.floor((N-3)/6))
num_group = 3
N_per_group = np.array([T*2 + 1, T*2 + 1 , N - (T*2 + 1) * (num_group-1)])




# learning parameters
max_iter = 50
np.random.seed(42) # set the seed of the random number generator for consistency

p, q_bit_X, q_bit_y = 2^26 - 5, 2, 0  # field size

alpha_exp = 10
coeffs0_exp = 1
coeffs1_exp = 11
trunc_scale = alpha_exp + coeffs1_exp - q_bit_y
trunc_k, trunc_m = 24, trunc_scale




# For debugging
Load_images_ON = 1      # 0: using synthetic data, 1: Load CIFAR-10 dataset
m_syn,d_syn = 90, 28*28
DEBUG_MODE = [0,0,0,0,0]



if rank == 0:
    print 'Hi from crypto-service provider,', 'rank',rank

    if Load_images_ON:
        t0_read = time.time() # start timer

        print 'crypto-service provider is reading dataset' 

        print("00. Load Gisette")
        dir = './datasets/Gisette/'
        f = open(dir+'gisette_train.data')
        data=[]
        for row in f.readlines():
            data.append((row.strip()).split(" "))
        f.close()

        f= open(dir+"gisette_train.labels")
        classes=[]
        for row in f.readlines():
            classes.append((row.strip()).split(" "))
        f.close()

        X_train, y_train = np.array(data).astype(int), np.array(classes).astype(int)

        f = open(dir+'gisette_valid.data')
        data=[]
        for row in f.readlines():
            data.append((row.strip()).split(" "))
        f.close()

        f= open(dir+"gisette_valid.labels")
        classes=[]
        for row in f.readlines():
            classes.append((row.strip()).split(" "))
        f.close()

        X_test, y_test = np.array(data).astype(int), np.array(classes).astype(int)

        print 'Dim: X_train, y_train = ',np.shape(X_train),np.shape(y_train)
        print 'Dim: X_test, y_test = ',np.shape(X_test),np.shape(y_test)

        t_read = time.time() - t0_read     
        print('Time spent for reading dataset (sec)', t_read)
        print

        m, d = np.shape(X_train)
        X_train = X_train/np.max(X_train)
        X_test  = X_test/np.max(X_test)

        y_test = (y_test + 1)/2
        y_train = (y_train+1)/2
        
        X_tmp = X_train

        m_group = X_tmp.shape[0]/3
        m = m_group * 3
        d = X_tmp.shape[1]
        
        X = X_tmp[0:m,:]
        y = np.reshape(y_train[0:m], (m, 1)) # reshape row vector into a column vector 
        y_test = np.reshape(y_test, (len(y_test),1))

        # release the memory
        X_train = None
        y_train = None
        X_tmp = None

        gc.collect()

    else: # for speeding up the set X (instead of loading images)
        print 'crypto-service provider generating dataset'
        m = m_syn
        d = d_syn
        train_images = np.random.randn(m, d)
        train_labels = np.random.randint(2,size=(m,1)) 
    
        X = train_images[:m] # extract the first m rows 
        y = train_labels[:m] # extract the first m elements

        m_group = m / 3
    
    
    t0_preprocessing = time.time()

    print('01.Data conversion: real to finite field')
    t0_q = time.time()
    X_q = my_q(X,q_bit_X,p) # X_q: matrix with size ( m by d )

    t_q = time.time() - t0_q # time spent in reading dataset

    y_scale = ((2**q_bit_y) * y).astype('int64')


    print '\t\t time to compute my_q:',t_q

    # initialize model parameters
    w = (1/float(m))*np.random.rand(d,1) 
    w_q_tmp = my_q(w, q_bit_w, p)

    print('02. Secret Shares generation in finite field')
    #X_SS_T = BGW_encoding(X_q,N,T,p)
    X_SS_T = np.empty((N,m_group,d),dtype='int64')
    w_SS_T = np.empty((N,d,1)  ,dtype='int64')

    N_idx = 0
    for i in range(num_group):
        N_cur_group = N_per_group[i]
        temp = BGW_encoding(X_q[m_group*i:m_group*(i+1),:],N_cur_group,T,p)
        X_SS_T[N_idx:N_idx+N_cur_group,:,:] = temp

        temp_w = BGW_encoding(w_q_tmp,N_cur_group,T,p)
        w_SS_T[N_idx:N_idx+N_cur_group,:,:] = temp_w

    for j in range(1, N+1):
#         print('Sending data to worker', j)
       
        data_X_T    = np.reshape(X_SS_T[j-1,:,:], d * m_group) # send data in vector format
        group_idx = (j-1)/m_group
        data_y      = np.reshape(y_scale[m_group*group_idx:m_group*(group_idx+1)], m_group) # send data in vector format
        data_w_T    = np.reshape(w_SS_T[j-1,:,:], d) # send data in vector format

        comm.send(m_group, dest=j) # send number of rows =  number of training samples
        comm.send(d, dest=j) # send number of columns = number of features
        comm.Send(data_X_T, dest=j) # sent data to worker j
        comm.Send(data_y, dest=j) # sent data to worker j
        comm.Send(data_w_T, dest=j) # sent data to worker j

    X_SS_T = None
    w_SS_T = None
    data_X_T,data_y,data_w_T = None, None, None
    gc.collect()

    print('03. Random matrix and corresponding SS generation')
    #r_mult1 = np.random.randint(p, size=(m,1))

    r1 = np.random.randint(2**trunc_m,          size=(d,1))
    r2 = np.random.randint(2**(trunc_k-trunc_m),size=(d,1))

    r1_BGW = np.empty((N,d,1)  ,dtype='int64')
    r2_BGW = np.empty((N,d,1)  ,dtype='int64')
    
    N_idx = 0
    for i in range(num_group):
        N_cur_group = N_per_group[i]

        temp_r1 = BGW_encoding(r1,N_cur_group,T,p)
        temp_r2 = BGW_encoding(r2,N_cur_group,T,p)
        r1_BGW[N_idx:N_idx+N_cur_group,:,:] = temp_r1
        r2_BGW[N_idx:N_idx+N_cur_group,:,:] = temp_r2

        N_idx = N_idx + N_cur_group

    t0_CSP_send_SS = time.time()

    for j in range(1, N+1):
#         print('Sending data to worker', j)
       
        #data_X_T    = np.reshape(X_SS_T[j-1,:,:], d * m_group) # send data in vector format
        #group_idx = (j-1)/m_group
        #data_y      = np.reshape(y_scale[m_group*group_idx:m_group*(group_idx+1)], m_group) # send data in vector format
        #data_w_T    = np.reshape(w_SS_T[j-1,:,:], d) # send data in vector format
        
        data_r1_T   = np.reshape(r1_BGW[j-1,:,:], d) # send data in vector format
        data_r2_T   = np.reshape(r2_BGW[j-1,:,:], d) # send data in vector format
        
        comm.Send(data_r1_T, dest=j) # sent data to worker j
        comm.Send(data_r2_T, dest=j) # sent data to worker j
    
    print('sending X_SS_T & random SS is done')

    t_CSP_send_SS =  time.time() - t0_CSP_send_SS
    t_preprocessing  =  time.time() - t0_preprocessing

    print '[ crypto-service provider ] sending X_SS_T & random SS is done' 
    print '[ crypto-service provider ] Preprocessing Time=', t_preprocessing,', sending SS in preprocessing=',t_CSP_send_SS 

    worker_idx = range(2*T+1)
    
    if DEBUG_MODE[0]:
        dec_input = np.empty((N, max_iter+1, d), dtype='int64')
        for j in range(1,N+1):
            data = np.empty((max_iter+1) * d, dtype='int64')
            comm.Recv(data, source=j)
            dec_input[j-1,:,:] = np.reshape(data,(max_iter+1, d))
    
    

        for i in range(max_iter + 1):
            w_dec = BGW_decoding(dec_input[worker_idx,i,:], worker_idx, p)
            w_tmp = my_q_inv(w_dec, 0,p)
            w_tmp = (2**(-q_bit_y))*np.reshape(w_tmp,(d,1))
        
            f = sigmoid(X.dot(w_tmp))
            error = -(1/float(len(y)))*(np.dot(y.T, np.log(f)) + np.dot(1-y.T, np.log(1-f)))
    
            print(i, error)
            #plt.title("field size = p ="+str(p))
            #plt.plot(w_tmp,'r')
            ## plt.plot(c0_my,'b')
            #plt.legend(['a/(2**m) w/o truncation', 'd/(2**m) w/ truncation'])
            #plt.show()
    
    if DEBUG_MODE[1]:
        dec_input = np.empty((N,d*d), dtype='int64')
        for j in range(1,N+1):
            comm.Recv(dec_input[j-1,:], source=j)
        XTX_dec = BGW_decoding(dec_input[worker_idx,:], worker_idx, p)

        answer = np.mod(X_q[0:m_group,:].transpose().dot(X_q[0:m_group,:]),p)
        answer = np.reshape(answer, (1,d*d))
        print 'XTX_dec:', XTX_dec.shape, XTX_dec[0,350:353]
        print answer[0,350:353]
    
    if DEBUG_MODE[2]:
        dec_input = np.empty((N,d), dtype='int64')
        for j in range(1,N+1):
            comm.Recv(dec_input[j-1,:], source=j)
        trunc_dec = BGW_decoding(dec_input[worker_idx,:], worker_idx, p)
        print 'w:', trunc_dec.shape, trunc_dec[0,350:353]

    if DEBUG_MODE[3]:
        dec_input = np.empty((N,d), dtype='int64')
        for j in range(1,N+1):
            comm.Recv(dec_input[j-1,:], source=j)
        trunc_dec = BGW_decoding(dec_input[worker_idx,:], worker_idx, p)
        print 'XTXw:', trunc_dec.shape, trunc_dec[0,350:353]
        #print dec_input[0,0:3]

    if DEBUG_MODE[4]:
        dec_input = np.empty((N,d), dtype='int64')
        for j in range(1,N+1):
            comm.Recv(dec_input[j-1,:], source=j)
        trunc_dec = BGW_decoding(dec_input[worker_idx,:], worker_idx, p)
        print 'af trunc:', trunc_dec.shape, trunc_dec[0,350:353]
    

    
    N_time_set = 6
    time_set_workers = np.empty((N,N_time_set), dtype='float')
    for j in range(1,N+1):
        comm.Recv(time_set_workers[j-1,:], source=j)
    #print(time_set)

    total_time = time.time() - t0_preprocessing
    print('[ crypto-service provider ] Total training time = ', total_time)

    time_out = []

    time_set = {'K': K,
                'T': T,
                'total_time': total_time,
                't_CSP_send_SS': t_CSP_send_SS,
                't_preprocessing': t_preprocessing,
                'time_set_workers': time_set_workers}
    time_out.append(time_set)

    
    pickle.dump(time_out, open('BGW_GISETTE_'+str(N), 'wb'), -1)



    T_workers = np.sum(time_set_workers, axis=0)/N
    #print T_workers
    # time_set = np.array([t_mult1, t_mult2, t_trunc, t_XTX, t_preprocessing, t_mainloop])

    print ' total iter time', np.sum(T_workers[:4])
    print
    print 'preprocessing @ wokers=', T_workers[4]
    print 'Main Loop     @ wokers=', T_workers[5]
    print 
    print 'N,K,T',N,K,T
    


elif rank <= N:
    def matrix_multiply(X1, X2, p, T, rank, user_list=None): # secure matrix multiplication (single reshare for degree reduction)
        
        if not user_list:
            my_user_list = range(1, 2*T+2)
        else: 
            my_user_list = user_list

        t0_matmul = time.time()

        mult = np.mod(X1.dot(X2), p) # multiply the current secret shares
        
        t_matmul = time.time() - t0_matmul

        # degree reduction
        t0 = time.time()

        row, col  = np.shape(mult)
        
        Q = np.empty((len(my_user_list), row, col), dtype='int64') # store secret shares        
        
        for j in my_user_list: # workers 1 to 2T+1 sends shares
     
            if rank == j:
            
                Qss = BGW_encoding(mult,len(my_user_list),T,p) 
                Q[j-my_user_list[0], :, :] = Qss[j-my_user_list[0], :, :] # own secret share
                
                #for i in range(1, rank) +  range(rank+1,N+1): # secret share q
                #    data = np.reshape(Qss[j-my_user_list[0],:,:],row*col)
                #    comm.Send(data, dest=i) # sent data to worker j
                for i in my_user_list:
                    if i != rank:
                        data = np.reshape(Qss[i-my_user_list[0],:,:],row*col)
                        comm.Send(data, dest=i) # sent data to worker j
            else:
                data = np.empty(row*col, dtype='int64')
                comm.Recv(data, source=j)
                Q[j-my_user_list[0],:,:] = np.reshape(data,(row,col)) # coefficients for the polynomial 
        
        Lcoeffs = gen_BGW_lambda_s(range(1, 2*T+2), p) 
        
        out = np.empty((row, col), dtype='int64')
                   
        for j in range(2*T+1):        
            out += Lcoeffs[0][j] * Q[j,:,:] # index 0 to 2T+1, XXX MAKE THIS FINITE FIELD OPERATION WITH MOD
        
        t_deg_reduction = time.time() - t0
        return np.mod(out, p), t_matmul, t_deg_reduction

    def MPI_MultPassive(A_SS_T, B_SS_T, R_SS_T, R_SS_2T, N,T,p, rank, user_list):
        t0 = time.time()
        #AB_SS_2T = np.mod(A_SS_T.astype('int64').dot(B_SS_T.astype('int64')), p)
        AB_SS_2T = np.mod(A_SS_T.dot(B_SS_T), p)
        t01 = time.time()
        delta_SS_2T = np.mod(AB_SS_2T - R_SS_2T,p)

        delta_size = np.prod(delta_SS_2T.shape)
        delta_SS_2T = np.reshape(delta_SS_2T,(1,delta_size)) 
        
        t1 = time.time()
        
        my_user_list = user_list

        dec_input = np.empty((len(my_user_list), delta_size), dtype='int64')
        
        for j in my_user_list:
            if rank == j:
                dec_input[j-my_user_list[0],:] = delta_SS_2T[0][:]
                for i in my_user_list: # secret share q
                    if i != rank:
                        data = np.reshape(delta_SS_2T[0][:], delta_size)
                        comm.Send(data, dest=i) # sent data to worker j
                #for 
            else:
                data = np.empty(delta_size, dtype='int64')
                comm.Recv(data, source=j)
                dec_input[j-my_user_list[0],:] = data # coefficients for the polynomial
        
        t2 = time.time()
        delta = BGW_decoding(dec_input[0:2*T+1,:], range(2*T+1), p) 
        # print 'rank=',rank, dec_input[:,0], delta
        delta = np.reshape(delta, AB_SS_2T.shape)

        t3 = time.time()
        time_set = np.array([t01-t0, t1-t01, t2-t1, t3-t2])
        #print 'Mult gate info:', time_set
        return np.mod(delta + R_SS_T, p).astype('int64')

    def MPI_TruncPr(in_SS_T, r1_SS_T, r2_SS_T, trunc_k, trunc_m, T, p, rank, user_list):
        my_user_list = user_list

        a_SS_T = in_SS_T.astype('int64')
        trunc_size = np.prod(a_SS_T.shape)

        a_SS_T = np.reshape(a_SS_T,trunc_size)
        r1_SS_T = np.reshape(r1_SS_T,trunc_size)
        r2_SS_T = np.reshape(r2_SS_T,trunc_size)    

        b_SS_T = np.mod(a_SS_T + 2**(trunc_k-1), p)

        r_SS_T = np.mod((2**trunc_m)*r2_SS_T + r1_SS_T , p)

        c_SS_T = np.mod( b_SS_T + r_SS_T ,p)
        # print 'rank=',rank, c_SS_T.shape
    
        dec_input = np.empty((T+1, trunc_size), dtype='int64')
        
        for j in my_user_list[0:T+1]:
            if rank == j:
                dec_input[j-my_user_list[0],:] = c_SS_T
                for i in my_user_list: # secret share q
                    if i != rank:
                        data = c_SS_T
                        comm.Send(data, dest=i) # sent data to worker j
            else:
                data = np.empty(trunc_size, dtype='int64')
                comm.Recv(data, source=j)
                dec_input[j-my_user_list[0],:] = data # coefficients for the polynomial
        
        c_dec = BGW_decoding(dec_input, range(T+1), p) 
        # print 'rank=',rank, 'c_dec is completed', c_dec.shape
        c_prime = np.mod( np.reshape(c_dec, trunc_size), 2**trunc_m )

        a_prime_SS_T = np.mod(c_prime - r1_SS_T, p)

        d_SS_T = np.mod(a_SS_T- a_prime_SS_T, p)
    
        d_SS_T = divmod(d_SS_T, 2**trunc_m, p)

        d_SS_T = np.reshape(d_SS_T, in_SS_T.shape)

        return d_SS_T.astype('int64')

    
    ######### end of function definition
    print 'Hi from worker,', 'rank',rank

    m = comm.recv(source=0) # number of rows =  number of training samples
    d = comm.recv(source=0) # number of columns  = number of features
    
    data = np.empty(m*d, dtype='int64')  # allocate space to receive the matrix
    comm.Recv(data, source=0)
    X_SS_T = np.reshape(data, (m, d)) # coded matrix

    data = np.empty(m*1, dtype='int64')  # allocate space to receive the matrix
    comm.Recv(data, source=0)
    y_scale = np.reshape(data, (m, 1)) # coded matrix

    data = np.empty(d*1, dtype='int64')  # allocate space to receive the matrix
    comm.Recv(data, source=0)
    w_SS_T = np.reshape(data, (d, 1)) # coded matrix

    data = np.empty(d*1, dtype='int64')  # allocate space to receive the matrix
    comm.Recv(data, source=0)
    r1_SS_T = np.reshape(data, (d, 1)) # coded matrix

    data = np.empty(d*1, dtype='int64')  # allocate space to receive the matrix
    comm.Recv(data, source=0)
    r2_SS_T = np.reshape(data, (d, 1)) # coded matrix

    my_group = min((rank-1)/(2*T+1),2)
    group_stt_info = [1,2*T+2,4*T+3,N+1]
    user_list = range(group_stt_info[my_group],group_stt_info[my_group+1])

    my_offset = np.mod((rank-1), 2*T+1) + 1 
    user_list2 = [my_offset, my_offset+2*T+1, my_offset+4*T+2]
    
    # preprocessing XTX
    t0_XTX = time.time()
    XTX_SS_T, t_XTX_, t_deg_reduction_XTX = matrix_multiply(X_SS_T.T, X_SS_T, p, T, rank, user_list) 
    t_XTX = time.time() - t0_XTX

    print 'XTX info :',XTX_SS_T.shape, t_XTX

    
    c0_my = np.int64(2**(q_bit_y + coeffs1_exp-coeffs0_exp) - (2**coeffs1_exp) * y_scale)
    tmp_SS_T = np.mod(np.dot(X_SS_T.T, c0_my), p)
    t_preprocessing = time.time() - t0_XTX

    # set parameters
    iter = 0
    hist_w_SS_T = np.empty((max_iter+1,d), dtype='int64')
    hist_w_SS_T[0,:] = np.reshape(w_SS_T, d)
    t_mult1, t_mult2, t_trunc = 0, 0, 0
    t0_mainloop = time.time()
    while (iter < max_iter):
    
        iter = iter + 1
        #print('iter=',iter)

        t0_mult1 = time.time()      
        XTXw_SS_T, t_XTXw, t_dr_tmp = matrix_multiply(XTX_SS_T, w_SS_T, p, T, rank, user_list)
        XTXw_SS_T = np.reshape(XTXw_SS_T, (d,1)) #XTXw_SS_T = np.zeros((d,1),dtype='int64')
        
    
        #print('y_scale',y_scale.shape, y_scale[350:353])
        #print('c0_my',c0_my.shape, c0_my[350:353])

        #tmp_SS_T = np.mod(c0_my + Xw_SS_T, p)
        #print(Xw_SS_T.shape, tmp_SS_T.shape)
    
        grad_SS_T = np.mod(XTXw_SS_T + tmp_SS_T, p)
        #grad_SS_T = np.mod(XTXw_SS_T , p)

        t_mult1 += time.time() - t0_mult1

        t0_trunc = time.time()
        grad_trunc_SS_T = MPI_TruncPr(grad_SS_T, r1_SS_T, r2_SS_T, trunc_k, trunc_scale, T, p, rank, user_list)
        t_trunc += time.time() - t0_trunc

        if rank in user_list2:
            rv_grad_SS_T = np.zeros(d,dtype='int64')
            for j in user_list2:
                if rank == j:
                    for i in user_list2:
                        if i != rank:
                            comm.Send(grad_trunc_SS_T[:], dest=i)
                else:
                    data = np.empty(d, dtype='int64')
                    comm.Recv(data, source=j)
                    rv_grad_SS_T = np.mod(rv_grad_SS_T + data, p)

            rv_grad_SS_T = np.reshape(rv_grad_SS_T,(d,1))
            grad_trunc_SS_T = np.mod(grad_trunc_SS_T+rv_grad_SS_T,p)


        w_SS_T = np.mod(w_SS_T - grad_trunc_SS_T, p)
        #print 'trunc_info:', trunc_k, trunc_scale
        #print(grad_SS_T.shape, grad_trunc_SS_T.shape)

        hist_w_SS_T[iter,:] = np.reshape(w_SS_T, d)
    
    t_mainloop = time.time() - t0_mainloop
    
    if DEBUG_MODE[0]:
        data = np.reshape(hist_w_SS_T, (max_iter+1) * d)
        comm.Send(data, dest=0)
    if DEBUG_MODE[1]:
        data = np.reshape(XTX_SS_T, d*d)
        comm.Send(data, dest=0)
    if DEBUG_MODE[2]:
        data = np.reshape(w_SS_T, d)
        comm.Send(data, dest=0)
    if DEBUG_MODE[3]:
        data = np.reshape(XTXw_SS_T, d)
        comm.Send(data, dest=0)
    if DEBUG_MODE[4]:
        data = np.reshape(grad_trunc_SS_T, d)
        comm.Send(data, dest=0)
    
    

    time_set = np.array([t_mult1, t_mult2, t_trunc, t_XTX, t_preprocessing, t_mainloop])
    #print 'time_set shape:', time_set.shape
    comm.Send(time_set, dest=0)
    

    