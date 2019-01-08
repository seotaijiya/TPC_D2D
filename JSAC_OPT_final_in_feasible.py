import tensorflow as tf
import numpy as np
import math
import os
import time
import pandas as pd
from numpy import linalg as LA


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


''' WMMSE results

'''

def WMMSE_sum_rate(p_int, H, Pmax, int_cell):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / ( np.matmul(np.square(H[i, :]), np.square(b)) + int_cell[i])
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + np.log(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = np.minimum(btmp, np.sqrt(Pmax)) + np.maximum(btmp, 0) - btmp
        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ( np.matmul(np.square(H[i, :]), np.square(b) ) + int_cell[i])
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i] + 1e-12)
            vnew = vnew + np.log(w[i])
        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break
    p_opt = np.square(b)
    return p_opt


## Calculate SINR, interference, out_prob for given values
def cal_performance(tx_pow, inter_threshold, num_band, ch_val):
    SINR_mat = []
    out_prob_mat = []
    int_CUE_mat = []
    for i in range(num_band):
        ch_val_band = np.array(ch_val[i], copy=True)
        ch_w = np.multiply(tx_pow[i], ch_val_band)
        sig = np.array(np.diag(ch_w), copy=True)
        int = np.sum(ch_w, 1)
        int_2 = np.sum(np.transpose(ch_w), 1)
        cap = np.divide(sig, int - sig + N0W)
        out_prob = np.mean((int - sig)[-1:] > inter_threshold)
        int_CUE = (int - sig)[-1:]
        SINR_mat.append(cap)
        int_CUE_mat.append(int_CUE)
        out_prob_mat.append(out_prob)

    return SINR_mat, int_CUE_mat, out_prob_mat


## Calculate SINR of eq given channel pt and num d2d
## Assume that the PU use the maximum transmit power
def sinr_eq(p_t, inter_threshold, num_d2d, num_band, ch_val, rate_thr):
    tx_pow = 1.0/num_band * p_t * np.ones((num_band, num_d2d+1))
    tx_pow[:,-1:] = p_t
    SINR_mat, int_CUE_mat, out_prob_mat = cal_performance(tx_pow, inter_threshold, num_band, ch_val)
    return_cap = np.sum(np.log2(1+np.array(SINR_mat)), 0)
    return_CUI = np.mean(int_CUE_mat, 0)
    return_OUT_prob = np.mean(out_prob_mat, 0)
    return_tx_pow = tx_pow

    ##########################
    ## FIND whether the QoS constraint is satisfied
    ##########################
    rate_temp = return_cap[:-1]
    rate_satisfied = np.array(rate_temp) < rate_thr
    return_OUT_DUE = np.mean(rate_satisfied.astype(float))
    return return_cap, return_CUI, return_OUT_prob, return_OUT_DUE, return_tx_pow


## Calculate SINR of eq given channel pt and num d2d
def sinr_dnn(tx_pow, inter_threshold, num_d2d, num_band, ch_val, rate_thr):
    SINR_mat, int_CUE_mat, out_prob_mat = cal_performance(tx_pow, inter_threshold, num_band, ch_val)
    return_cap = np.sum(np.log2(1 + np.array(SINR_mat)), 0)
    return_CUI = np.mean(int_CUE_mat, 0)
    return_OUT_prob = np.mean(out_prob_mat, 0)
    return_tx_pow = tx_pow
    ##########################
    ## FIND whether the QoS constraint is satisfied
    ##########################
    rate_temp = return_cap[:-1]
    rate_satisfied = np.array(rate_temp) < rate_thr
    return_OUT_DUE = np.mean(rate_satisfied.astype(float))
    return return_cap, return_CUI, return_OUT_prob, return_OUT_DUE, return_tx_pow



## Calculate SINR of eq given channel pt and num d2d
def sinr_dnn_infeasible(tx_pow, inter_threshold, num_d2d, num_band, ch_val, rate_thr):
    SINR_mat, int_CUE_mat, out_prob_mat = cal_performance(tx_pow, inter_threshold, num_band, ch_val)
    return_cap = np.sum(np.log2(1 + np.array(SINR_mat)), 0)
    return_CUI = int_CUE_mat
    return_OUT_prob = np.mean(out_prob_mat, 0)
    return_tx_pow = tx_pow
    ##########################
    ## FIND whether the QoS constraint is satisfied
    ##########################
    rate_temp = return_cap[:-1]
    rate_satisfied = np.array(rate_temp) < rate_thr
    return_OUT_DUE = np.mean(rate_satisfied.astype(float))
    return return_cap, return_CUI, return_OUT_prob, return_OUT_DUE, return_tx_pow

def sinr_conv_opt(p_t, inter_threshold, num_d2d, num_band, ch_val):
    tx_pow = 1.0 / num_band * p_t * np.ones((num_band, num_d2d + 1))
    tx_pow[:, -1:] = p_t
    rate_cur = -1
    print_power = 0

    for i_1 in range(granu):
        for i_2 in range(granu):
            for i_3 in range(granu):
                for i_4 in range(granu):
                    # tx power of first
                    tx_pow[0, 0] = p_t * i_1/(granu-1)
                    tx_pow[0, 1] = p_t * i_2/(granu-1)
                    tx_pow[1, 0] = p_t * i_3/(granu-1)
                    tx_pow[1, 1] = p_t * i_4/(granu-1)


                    if np.any(tx_pow.sum(axis=0)[:-1] > p_t) == False:
                        SINR_mat_temp, int_CUE_mat_temp, out_prob_mat_temp = cal_performance(tx_pow, inter_threshold, num_band, ch_val)
                        #print("   ")
                        #print('int = ', np.log10(int_CUE_mat_temp))
                        #print('rate = ', np.log2(1+np.array(SINR_mat_temp)))
                        #print('threshodl = ', np.array(int_CUE_mat_temp)>inter_threshold)
                        if np.any(np.array(int_CUE_mat_temp)>inter_threshold) == False:
                            rate_temp = np.sum(np.log2(1 + np.array(SINR_mat_temp)[:,:-1]))
                            if rate_temp > rate_cur:
                                rate_cur = np.array(rate_temp, copy=True)
                                SINR_mat = np.array(SINR_mat_temp, copy=True)
                                int_CUE_mat = np.array(int_CUE_mat_temp, copy=True)
                                out_prob_mat = np.array(out_prob_mat_temp, copy=True)
                                print_power = np.array(tx_pow, copy=True)

    return np.sum(np.log2(1 + np.array(SINR_mat)), 0), np.mean(int_CUE_mat, 0), np.mean(out_prob_mat, 0), print_power

def sinr_conv_opt_ee(p_t, inter_threshold, num_d2d, num_band, ch_val):
    tx_pow = 1.0 / num_band * p_t * np.ones((num_band, num_d2d + 1))
    tx_pow[:, -1:] = p_t
    ee_cur = -1
    print_power = 0

    for i_1 in range(granu):
        for i_2 in range(granu):
            for i_3 in range(granu):
                for i_4 in range(granu):
                    # tx power of first
                    tx_pow[0, 0] = p_t * i_1/(granu-1)
                    tx_pow[0, 1] = p_t * i_2/(granu-1)
                    tx_pow[1, 0] = p_t * i_3/(granu-1)
                    tx_pow[1, 1] = p_t * i_4/(granu-1)


                    if np.any(tx_pow.sum(axis=0)[:-1] > p_t) == False:
                        SINR_mat_temp, int_CUE_mat_temp, out_prob_mat_temp = cal_performance(tx_pow, inter_threshold, num_band, ch_val)
                        if np.any(np.array(int_CUE_mat_temp)>inter_threshold) == False:
                            ee_temp = np.mean(np.divide(np.sum(np.log2(1 + np.array(SINR_mat_temp)[:, :-1]), 0),
                                                        (np.sum(tx_pow, 0)[:-1] + p_c)))
                            if ee_temp > ee_cur:
                                ee_cur = np.array(ee_temp, copy=True)
                                SINR_mat = np.array(SINR_mat_temp, copy=True)
                                int_CUE_mat = np.array(int_CUE_mat_temp, copy=True)
                                out_prob_mat = np.array(out_prob_mat_temp, copy=True)
                                print_power = np.array(tx_pow, copy=True)

    return np.sum(np.log2(1 + np.array(SINR_mat)), 0), np.mean(int_CUE_mat, 0), np.mean(out_prob_mat, 0), print_power

#################################################################################
### Aggretate all conventional schemes - Rate maximize, EE maximize, TX minimize
#################################################################################
def sinr_conv_opt_all(p_t, inter_threshold, num_d2d, num_band, ch_val, rate_thr):
    ### Initialize tx power
    tx_pow = 1.0 / num_band * p_t * np.ones((num_band, num_d2d + 1))
    tx_pow[:, -1:] = p_t

    ## Initial value of tx power, rate, ee
    tx_pow_cur = 100000000
    rate_cur = -1
    ee_cur = -1
    feasible_result = 0

    ###################################
    ### Initialization for returned valules
    ##################################
    SINR_rate = np.zeros((2, 3))
    int_CUE_rate = np.zeros((2, 1))
    out_prob_rate = np.zeros((1, 1))
    tx_power_rate = np.zeros((2, 3))
    out_DUE_rate = np.zeros((1, 1))

    SINR_ee = np.zeros((2, 3))
    int_CUE_ee = np.zeros((2, 1))
    out_prob_ee = np.zeros((1, 1))
    tx_power_ee = np.zeros((2, 3))
    out_DUE_ee = np.zeros((1, 1))

    SINR_tx = np.zeros((2, 3))
    int_CUE_tx = np.zeros((2, 1))
    out_prob_tx = np.zeros((1, 1))
    tx_power_tx = np.zeros((2, 3))
    out_DUE_tx = np.zeros((1, 1))


    for i_1 in range(granu):
        for i_2 in range(granu):
            for i_3 in range(granu):
                for i_4 in range(granu):
                    # tx power of first
                    tx_pow[0, 0] = p_t * i_1/(granu-1)
                    tx_pow[0, 1] = p_t * i_2/(granu-1)
                    tx_pow[1, 0] = p_t * i_3/(granu-1)
                    tx_pow[1, 1] = p_t * i_4/(granu-1)
                    if np.any(tx_pow.sum(axis=0)[:-1] > p_t) == False:
                        SINR_mat_temp, int_CUE_mat_temp, out_prob_mat_temp = cal_performance(tx_pow, inter_threshold, num_band, ch_val)
                        if np.any(np.array(int_CUE_mat_temp)>inter_threshold) == False:
                            rate_temp = np.sum(np.log2(1 + np.array(SINR_mat_temp)), 0)[:-1]
                            if np.all(rate_temp>rate_thr) == True:
                                ee_temp = np.mean(np.divide(np.sum(np.log2(1 + np.array(SINR_mat_temp)[:, :-1]), 0),
                                                            (np.sum(tx_pow, 0)[:-1] + p_c)))
                                tx_pow_temp = tx_pow[0, 0] + tx_pow[0, 1] + tx_pow[1, 0] + tx_pow[1, 1]
                                feasible_result = 1
                                ###########################
                                ### Maximize of RATE ######
                                ###########################
                                if np.sum(rate_temp) > rate_cur:
                                    rate_cur = np.array(np.sum(rate_temp), copy=True)
                                    SINR_rate = np.array(SINR_mat_temp, copy=True)
                                    int_CUE_rate = np.array(int_CUE_mat_temp, copy=True)
                                    out_prob_rate = np.array(out_prob_mat_temp, copy=True)
                                    tx_power_rate = np.array(tx_pow, copy=True)
                                    out_DUE_rate = np.zeros((1, 1))

                                ###########################
                                ### Maximize of EE ######
                                ###########################
                                if ee_temp > ee_cur:
                                    ee_cur = np.array(ee_temp, copy=True)
                                    SINR_ee = np.array(SINR_mat_temp, copy=True)
                                    int_CUE_ee = np.array(int_CUE_mat_temp, copy=True)
                                    out_prob_ee = np.array(out_prob_mat_temp, copy=True)
                                    tx_power_ee = np.array(tx_pow, copy=True)
                                    out_DUE_ee = np.zeros((1, 1))

                                ###########################
                                ### Tx minimize ######
                                ###########################
                                if tx_pow_cur > tx_pow_temp:
                                    tx_pow_cur = np.array(tx_pow_temp, copy=True)
                                    SINR_tx = np.array(SINR_mat_temp, copy=True)
                                    int_CUE_tx = np.array(int_CUE_mat_temp, copy=True)
                                    out_prob_tx = np.array(out_prob_mat_temp, copy=True)
                                    tx_power_tx = np.array(tx_pow, copy=True)
                                    out_DUE_tx = np.zeros((1, 1))

    ########
    ## Initialize return value
    ########
    return_val_rate = []
    return_val_ee = []
    return_val_tx = []
    return_val_tot = []

    return_val_rate.append(np.sum(np.log2(1 + np.array(SINR_rate)), 0))
    return_val_ee.append(np.sum(np.log2(1 + np.array(SINR_ee)), 0))
    return_val_tx.append(np.sum(np.log2(1 + np.array(SINR_tx)), 0))

    return_val_rate.append(np.mean(int_CUE_rate, 0))
    return_val_ee.append(np.mean(int_CUE_ee, 0))
    return_val_tx.append(np.mean(int_CUE_tx, 0))

    return_val_rate.append(np.mean(out_prob_rate, 0))
    return_val_ee.append(np.mean(out_prob_ee, 0))
    return_val_tx.append(np.mean(out_prob_tx, 0))

    return_val_rate.append(np.mean(out_DUE_rate, 0))
    return_val_ee.append(np.mean(out_DUE_ee, 0))
    return_val_tx.append(np.mean(out_DUE_tx, 0))

    return_val_rate.append(tx_power_rate)
    return_val_ee.append(tx_power_ee)
    return_val_tx.append(tx_power_tx)

    return_val_tot.append(return_val_rate)
    return_val_tot.append(return_val_ee)
    return_val_tot.append(return_val_tx)


    return return_val_tot, feasible_result


'''
    Building DNN model
'''
def model(X, w1, w2, w3, w4, w5, w1_1, w2_1, w3_1, w4_1, w5_1, wo, wp, b1, b2, b3, b4, b5, b1_1, b2_1, b3_1, b4_1, b5_1,bo, bp, p_keep_conv, num_d2d, num_band):
    # Determines
    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)
    l5 = tf.nn.dropout(l5, p_keep_conv)

    ## lra represents the resource allocation for each band
    lra = tf.matmul(l5, wo) + bo
    lra = tf.nn.dropout(lra, p_keep_conv)


    l1_1 = tf.nn.relu(tf.matmul(X, w1_1) + b1_1)
    l1_1 = tf.nn.dropout(l1_1, p_keep_conv)

    l2_1 = tf.nn.relu(tf.matmul(l1_1, w2_1) + b2_1)
    l2_1 = tf.nn.dropout(l2_1, p_keep_conv)

    l3_1 = tf.nn.relu(tf.matmul(l2_1, w3_1) + b3_1)
    l3_1 = tf.nn.dropout(l3_1, p_keep_conv)

    l4_1 = tf.nn.relu(tf.matmul(l3_1, w4_1) + b4_1)
    l4_1 = tf.nn.dropout(l4_1, p_keep_conv)

    l5_1 = tf.nn.relu(tf.matmul(l4_1, w5_1) + b5_1)
    l5_1 = tf.nn.dropout(l5_1, p_keep_conv)

    ## lp represents the transmit power of users
    lp = tf.matmul(l5_1, wp) + bp
    lp = tf.nn.dropout(lp, p_keep_conv)


    ## Calculate the resource allocation for each users
    lo_resource_alloc = tf.reshape(lra, [-1, num_d2d, num_band])


    #lo_resource_arg = tf.argmax(lo_resource_alloc, 2)
    #lo_temp = tf.one_hot(lo_resource_arg, depth=num_band)

    lo_temp = tf.nn.softmax(lo_resource_alloc)

    ## Calculate the resource allocation for each users
    lo_power_alloc = tf.reshape(lp, [-1, num_d2d, 1])

    pyx = tf.multiply(lo_temp , tf.nn.sigmoid(lo_power_alloc))

    return pyx


'''
    Initialization of location information
'''
def loc_init(size_area, d2d_dist, num_d2d):
    rx_loc = size_area * (np.random.rand(num_d2d + 1, 2) - 0.5)
    tx_loc = np.zeros((num_d2d + 1, 2))
    for i in range(num_d2d+1):
        temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
        temp_chan = rx_loc[i, :] + temp_dist
        while (np.max(abs(temp_chan)) > size_area / 2) | (np.linalg.norm(temp_dist) > d2d_dist):
            temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
            temp_chan = rx_loc[i, :] + temp_dist
        tx_loc[i, :] = temp_chan
    return rx_loc, tx_loc


'''
    For the returned matrix, pu_ch_gain[0, : ] indicates the channel of RX 1
'''
def ch_gen(size_area, d2d_dist, num_d2d, num_samples, num_band):
    ch_w_fading = []

    for i in range(num_samples):
        rx_loc, tx_loc = loc_init(size_area, d2d_dist, num_d2d)

        ch_w_temp_band = []

        for j in range(num_band):


            rx_loc_temp = size_area * (np.random.rand(1, 2) - 0.5)

            temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
            temp_chan = rx_loc_temp + temp_dist
            while (np.max(abs(temp_chan)) > size_area / 2) | (np.linalg.norm(temp_dist) > d2d_dist):
                temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
                temp_chan = rx_loc_temp + temp_dist

            rx_loc[num_d2d, :] = rx_loc_temp
            tx_loc[num_d2d, :] = temp_chan


            ## generate distance_vector
            dist_vec = rx_loc.reshape(num_d2d + 1, 1, 2) - tx_loc
            dist_vec = np.linalg.norm(dist_vec, axis=2)
            dist_vec = np.maximum(dist_vec, 1)

            # find path loss // shadowing is not considered
            pu_ch_gain_db = - pl_const - pl_alpha * np.log10(dist_vec)
            pu_ch_gain = 10 ** (pu_ch_gain_db / 10)

            multi_fading = 0.5 * np.random.randn(num_d2d+1, num_d2d+1) ** 2 + 0.5 * np.random.randn(num_d2d+1, num_d2d+1) ** 2

            final_ch = np.maximum(pu_ch_gain * multi_fading, np.exp(-30))

            ch_w_temp_band.append(final_ch)
        ch_w_fading.append(ch_w_temp_band)
    return np.array(ch_w_fading)


def ch_gen_test(size_area, d2d_dist, num_d2d, num_samples, num_band):
    ch_w_fading = []

    for i in range(num_samples):
        rx_loc, tx_loc = loc_init(size_area, d2d_dist, num_d2d)

        ch_w_temp_band = []

        for j in range(num_band):


            rx_loc_temp = size_area * (np.random.rand(1, 2) - 0.5)

            temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
            temp_chan = rx_loc_temp + temp_dist
            while (np.max(abs(temp_chan)) > size_area / 2) | (np.linalg.norm(temp_dist) > d2d_dist):
                temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
                temp_chan = rx_loc_temp + temp_dist

            rx_loc[num_d2d, :] = rx_loc_temp
            tx_loc[num_d2d, :] = temp_chan


            ## generate distance_vector
            dist_vec = rx_loc.reshape(num_d2d + 1, 1, 2) - tx_loc
            dist_vec = np.linalg.norm(dist_vec, axis=2)
            dist_vec = np.maximum(dist_vec, 1)

            # find path loss // shadowing is not considered
            pu_ch_gain_db = - pl_const_test - pl_alpha_test * np.log10(dist_vec)
            pu_ch_gain = 10 ** (pu_ch_gain_db / 10)

            multi_fading = 0.5 * np.random.randn(num_d2d+1, num_d2d+1) ** 2 + 0.5 * np.random.randn(num_d2d+1, num_d2d+1) ** 2

            final_ch = np.maximum(pu_ch_gain * multi_fading, np.exp(-30))

            ch_w_temp_band.append(final_ch)
        ch_w_fading.append(ch_w_temp_band)
    return np.array(ch_w_fading)


'''
    For the returned matrix, pu_ch_gain[0, : ] indicates the channel of RX 1
'''
def per_eval(batch_size, inter_threshold, num_band, rate_thr, learning_rate_init = 0.00001, target=0):
    cap_mat_te_DUE = np.zeros((5, 1))
    cap_mat_tr_DUE = np.zeros((5, 1))
    cap_mat_eq_DUE = np.zeros((5, 1))
    cap_mat_inter_DUE = np.zeros((5, 1))
    cap_mat_opt_rate_DUE = np.zeros((5, 1))
    cap_mat_opt_ee_DUE = np.zeros((5, 1))
    cap_mat_opt_tx_DUE = np.zeros((5, 1))

    cap_mat_te_CUE = np.zeros((5, 1))
    cap_mat_tr_CUE = np.zeros((5, 1))
    cap_mat_eq_CUE = np.zeros((5, 1))
    cap_mat_inter_CUE = np.zeros((5, 1))
    cap_mat_opt_rate_CUE = np.zeros((5, 1))
    cap_mat_opt_ee_CUE = np.zeros((5, 1))
    cap_mat_opt_tx_CUE = np.zeros((5, 1))

    out_mat_te = np.zeros((5, 1))
    out_mat_tr = np.zeros((5, 1))
    out_mat_eq = np.zeros((5, 1))
    out_mat_inter = np.zeros((5, 1))
    out_mat_opt_rate = np.zeros((5, 1))
    out_mat_opt_ee = np.zeros((5, 1))
    out_mat_opt_tx = np.zeros((5, 1))

    out_DUE_mat_te = np.zeros((5, 1))
    out_DUE_mat_tr = np.zeros((5, 1))
    out_DUE_mat_eq = np.zeros((5, 1))
    out_DUE_mat_inter = np.zeros((5, 1))
    out_DUE_mat_opt_rate = np.zeros((5, 1))
    out_DUE_mat_opt_ee = np.zeros((5, 1))
    out_DUE_mat_opt_tx = np.zeros((5, 1))

    inter_mat_te = np.zeros((5, 1)) + 1e-50
    inter_mat_tr = np.zeros((5, 1)) + 1e-50
    inter_mat_eq = np.zeros((5, 1)) + 1e-50
    inter_mat_inter = np.zeros((5, 1)) + 1e-50
    inter_mat_opt_rate = np.zeros((5, 1)) + 1e-50
    inter_mat_opt_ee = np.zeros((5, 1)) + 1e-50
    inter_mat_opt_tx = np.zeros((5, 1)) + 1e-50

    ee_mat_te = np.zeros((5, 1))
    ee_mat_tr = np.zeros((5, 1))
    ee_mat_eq = np.zeros((5, 1))
    ee_mat_inter = np.zeros((5, 1))
    ee_mat_opt_rate = np.zeros((5, 1))
    ee_mat_opt_ee = np.zeros((5, 1))
    ee_mat_opt_tx = np.zeros((5, 1))

    tx_mat_te = np.zeros((5, 1))
    tx_mat_tr = np.zeros((5, 1))
    tx_mat_eq = np.zeros((5, 1))
    tx_mat_inter = np.zeros((5, 1))
    tx_mat_opt_rate = np.zeros((5, 1))
    tx_mat_opt_ee = np.zeros((5, 1))
    tx_mat_opt_tx = np.zeros((5, 1))

    saver = tf.train.Saver()

    lam_1_val_mat = [1e3, 0.9, 0.92, 0.95, 0.99]
    lam_2_val_mat = [1e3, 0.9, 0.92, 0.95, 0.99]

    for k in range(5):
        print("iteration: ", k)
        size_area = 30.0+10*k
        lam_1_val = gamma_1
        lam_2_val = gamma_2
        learning_rate = learning_rate_init
        for l in range(iter_num):
            cap_te_DUE, cap_tr_DUE, cap_eq_DUE, cap_inter_DUE, cap_opt_rate_DUE, cap_opt_ee_DUE, cap_opt_tx_DUE = 0, 0, 0, 0, 0, 0, 0
            cap_te_CUE, cap_tr_CUE, cap_eq_CUE, cap_inter_CUE, cap_opt_rate_CUE, cap_opt_ee_CUE, cap_opt_tx_CUE = 0, 0, 0, 0, 0, 0, 0
            out_te, out_tr, out_eq, out_inter, out_opt_rate, out_opt_ee, out_opt_tx = 0, 0, 0, 0, 0, 0, 0
            out_DUE_te, out_DUE_tr, out_DUE_eq, out_DUE_inter, out_DUE_opt_rate, out_DUE_opt_ee, out_DUE_opt_tx = 0, 0, 0, 0, 0, 0, 0
            inter_te, inter_tr, inter_eq, inter_inter, inter_opt_rate, inter_opt_ee, inter_opt_tx = 1e-50, 1e-50, 1e-50, 1e-50, 1e-50, 1e-50, 1e-50
            tx_te, tx_tr, tx_eq, tx_inter, tx_opt_rate, tx_opt_ee, tx_opt_tx = 0, 0, 0, 0, 0, 0, 0
            ee_te, ee_tr, ee_eq, ee_inter, ee_opt_rate, ee_opt_ee, ee_opt_tx = 0, 0, 0, 0, 0, 0, 0
            power_1, power_2, power_3 = 0, 0, 0

            #### Reltaed to channel sample
            ## Generating channel values
            ch_val_tot_train = ch_gen(size_area, d2d_dist, num_d2d, num_samples, num_band)
            ch_val_tot_test = ch_gen_test(size_area, d2d_dist, num_d2d, test_size, num_band)
            ch_val_tot = np.concatenate((ch_val_tot_train, ch_val_tot_test))


            _sample_ch = np.log10(ch_val_tot)
            avg_val = np.mean(_sample_ch)
            std_val = np.sqrt(np.var(_sample_ch))

            _sample_ch = (_sample_ch - avg_val) / std_val

            sample_ch = np.array(_sample_ch, copy=True)

            ch_val = sample_ch[:num_samples]
            ch_val_test = sample_ch[num_samples:]



            ## Calculate the
            ch_val_diag = []
            for j_2 in range(len(ch_val)):
                ch_val_diag_band = []
                for j_3 in range(num_band):
                    sig_diag = np.array(np.diag(ch_val[j_2, j_3]), copy=True)
                    ch_val_diag_band.append(sig_diag)
                ch_val_diag.append(ch_val_diag_band)
            ch_val_diag = np.array(ch_val_diag)


            ch_val_test_diag = []
            for j_2 in range(len(ch_val_test)):
                ch_val_diag_band = []
                for j_3 in range(num_band):
                    sig_test_diag = np.array(np.diag(ch_val_test[j_2, j_3]), copy=True)
                    ch_val_diag_band.append(sig_test_diag)
                ch_val_test_diag.append(ch_val_diag_band)
            ch_val_test_diag = np.array(ch_val_test_diag)


            with tf.Session() as sess:
                tf.initialize_all_variables().run()

                ref_opt_rate_rate = 0
                ref_opt_rate_ee = 0
                ref_opt_rate_out = 0
                ref_opt_rate_out_DUE = 0
                ref_opt_rate_CUI = 1e-20
                ref_opt_rate_tx = 1e-20

                ref_opt_ee_rate = 0
                ref_opt_ee_ee = 0
                ref_opt_ee_out = 0
                ref_opt_ee_out_DUE = 0
                ref_opt_ee_CUI = 1e-20
                ref_opt_ee_tx = 1e-20

                ref_opt_tx_rate = 0
                ref_opt_tx_ee = 0
                ref_opt_tx_out = 0
                ref_opt_tx_out_DUE = 0
                ref_opt_tx_CUI = 1e-20
                ref_opt_tx_tx = 1e-20

                ref_eq_rate = 0
                ref_eq_ee = 0
                ref_eq_out = 0
                ref_eq_out_DUE = 0
                ref_eq_CUI = 1e-20
                ref_eq_tx = 1e-20

                ref_inter_rate = 0
                ref_inter_ee = 0
                ref_inter_out = 0
                ref_inter_out_DUE = 0
                ref_inter_CUI = 1e-20
                ref_inter_tx = 1e-20

                #############################
                ## If Reuse is 0, reuse the saved model for DNN. NO TRAINING
                ## If Reuse is 1, JUST INFERENCE
                ## Test_size is the number of samples to be examined in the training for comparison
                #############################
                if reuse == 0:
                    test_size_init = test_size
                    test_size_init = -1
                else:
                    test_size_init = -1

                ##############
                ## correct_ch_num is the number of feasible channel vaules
                correct_ch_num  = 0
                for j_2 in range(test_size_init):
                    if j_2%50 == 0:
                        print("Test phase = ", j_2)
                    ##############
                    ## Test for opt rate
                    ###############
                    conv_result, feasible_check = sinr_conv_opt_all(p_t, inter_threshold, num_d2d, num_band,
                                                    10 ** (avg_val + std_val * ch_val[j_2]), rate_thr)

                    temp_cap_opt_rate = conv_result[0][0]
                    temp_CUI_opt_rate = conv_result[0][1]
                    temp_OUT_prob_opt_rate = conv_result[0][2]
                    temp_OUT_DUE_opt_rate = conv_result[0][3]
                    temp_tx_pow_opt_rate  = conv_result[0][4]

                    ref_opt_rate_rate = ref_opt_rate_rate + np.mean(temp_cap_opt_rate[:-1])
                    ref_opt_rate_ee = ref_opt_rate_ee + np.mean(np.divide(temp_cap_opt_rate[:-1], (np.sum(temp_tx_pow_opt_rate, 0)[:-1]  + p_c)  ))
                    ref_opt_rate_out = ref_opt_rate_out + temp_OUT_prob_opt_rate
                    ref_opt_rate_out_DUE = ref_opt_rate_out_DUE + temp_OUT_DUE_opt_rate
                    ref_opt_rate_CUI = ref_opt_rate_CUI + temp_CUI_opt_rate
                    ref_opt_rate_tx = ref_opt_rate_tx + np.sum(np.sum(temp_tx_pow_opt_rate, 0)[:-1]) / num_d2d


                    #############################
                    ## correct_ch_num holds the number of channel samples which are feasible
                    ##############################
                    correct_ch_num = correct_ch_num + feasible_check


                    ##############
                    ## Test for opt ee
                    ###############3
                    temp_cap_opt_ee = conv_result[1][0]
                    temp_CUI_opt_ee = conv_result[1][1]
                    temp_OUT_prob_opt_ee = conv_result[1][2]
                    temp_OUT_DUE_opt_ee = conv_result[1][3]
                    temp_tx_pow_opt_ee  = conv_result[1][4]

                    ref_opt_ee_rate = ref_opt_ee_rate + np.mean(temp_cap_opt_ee[:-1])
                    ref_opt_ee_ee = ref_opt_ee_ee + np.mean(np.divide(temp_cap_opt_ee[:-1], (np.sum(temp_tx_pow_opt_ee, 0)[:-1]  + p_c)  ))
                    ref_opt_ee_out = ref_opt_ee_out + temp_OUT_prob_opt_ee
                    ref_opt_ee_out_DUE = ref_opt_ee_out_DUE + temp_OUT_DUE_opt_ee
                    ref_opt_ee_CUI = ref_opt_ee_CUI + temp_CUI_opt_ee
                    ref_opt_ee_tx = ref_opt_ee_tx + np.sum(np.sum(temp_tx_pow_opt_ee, 0)[:-1]) / num_d2d


                    ##############
                    ## Test for opt tx
                    ###############3
                    temp_cap_opt_tx = conv_result[2][0]
                    temp_CUI_opt_tx = conv_result[2][1]
                    temp_OUT_prob_opt_tx = conv_result[2][2]
                    temp_OUT_DUE_opt_tx = conv_result[2][3]
                    temp_tx_pow_opt_tx  = conv_result[2][4]

                    ref_opt_tx_rate = ref_opt_tx_rate + np.mean(temp_cap_opt_tx[:-1])
                    ref_opt_tx_ee = ref_opt_tx_ee + np.mean(np.divide(temp_cap_opt_tx[:-1], (np.sum(temp_tx_pow_opt_tx, 0)[:-1] + p_c)  ))
                    ref_opt_tx_out = ref_opt_tx_out + temp_OUT_prob_opt_tx
                    ref_opt_tx_out_DUE = ref_opt_tx_out_DUE + temp_OUT_DUE_opt_tx
                    ref_opt_tx_CUI = ref_opt_tx_CUI + temp_CUI_opt_tx
                    ref_opt_tx_tx = ref_opt_tx_tx + np.sum(np.sum(temp_tx_pow_opt_tx, 0)[:-1]) / num_d2d


                    ##############
                    ## Test for EQ
                    ###############3
                    cap_eq, CUI_eq, OUT_prob_eq, OUT_DUE_eq, tx_pow_eq = sinr_eq(p_t, inter_threshold, num_d2d, num_band, 10 ** (avg_val + std_val * ch_val[j_2]), rate_thr)
                    ref_eq_rate = ref_eq_rate + np.mean(cap_eq[:-1])/ test_size
                    ref_eq_ee = ref_eq_ee + np.mean(np.divide(cap_eq[:-1], (np.sum(tx_pow_eq, 0)[:-1]  + p_c)))/ test_size
                    ref_eq_out = ref_eq_out + OUT_prob_eq/ test_size
                    ref_eq_out_DUE = ref_eq_out_DUE + OUT_DUE_eq / test_size
                    ref_eq_CUI = ref_eq_CUI + CUI_eq/ test_size
                    ref_eq_tx = ref_eq_tx + np.sum(np.sum(tx_pow_eq, 0)[:-1]) / num_d2d


                correct_ch_num = np.maximum(correct_ch_num, 1)

                print("feasible percentageg = %0.0f" %(correct_ch_num*1.0/test_size*100))



                ref_opt_rate_rate = ref_opt_rate_rate / correct_ch_num
                ref_opt_rate_out = ref_opt_rate_out / correct_ch_num
                ref_opt_rate_out_DUE = ref_opt_rate_out_DUE / correct_ch_num
                ref_opt_rate_CUI = ref_opt_rate_CUI / correct_ch_num
                ref_opt_rate_tx = ref_opt_rate_tx / correct_ch_num

                ref_opt_ee_rate = ref_opt_ee_rate / correct_ch_num
                ref_opt_ee_out = ref_opt_ee_out / correct_ch_num
                ref_opt_ee_out_DUE = ref_opt_ee_out_DUE / correct_ch_num
                ref_opt_ee_CUI = ref_opt_ee_CUI / correct_ch_num
                ref_opt_ee_tx = ref_opt_ee_tx / correct_ch_num

                ref_opt_tx_rate = ref_opt_tx_rate / correct_ch_num
                ref_opt_tx_out = ref_opt_tx_out / correct_ch_num
                ref_opt_tx_out_DUE = ref_opt_tx_out_DUE / correct_ch_num
                ref_opt_tx_CUI = ref_opt_tx_CUI / correct_ch_num
                ref_opt_tx_tx = ref_opt_tx_tx / correct_ch_num

                ##########################################
                ### If reuse == 0, restore the saved model
                ##########################################
                if reuse != 0:
                    tot_epoch = -1
                else:
                    tot_epoch = tot_epoch_real

                ##########################################
                ### If target = 0  => RATE MAXIMIZATION
                ### If target = 1  => EE MAXIMIZATION
                ### If target = 2  => TX MINIMIZATION
                ##########################################
                if target == 0:
                    opt_target = train_op_rate
                    cost_target = cost_rate
                elif target == 1:
                    opt_target = train_op_ee
                    cost_target = cost_ee
                else:
                    opt_target = train_op_tx
                    cost_target = cost_tx



                ## Initialize the previousr results for EE
                a_prev, b_prev, c_prev, d_prev, e_prev, f_prev, g_prev = 0, 0, 0, 0, 0, 0, 0

                for i in range(tot_epoch):
                    if i%200 == 0:
                        lam_1_val = lam_1_val*2
                        lam_2_val = lam_2_val*2
                        learning_rate = learning_rate / 1.5
                        print("update gamma = ", lam_1_val)
                    rand_perm = np.random.permutation(len(ch_val))
                    ch_val[:] = ch_val[rand_perm]
                    ch_val_diag[:] = ch_val_diag[rand_perm]

                    for start, end in zip(range(0, len(ch_val), batch_size), range(batch_size, len(ch_val), batch_size)):
                        feed_vec = ch_val[start:end]
                        feed_diag = ch_val_diag[start:end]
                        feed_vec = feed_vec.reshape(-1, (num_d2d+1)**2*num_band)

                        #################################
                        ### Important part - Training
                        ################################
                        sess.run(opt_target,
                                 feed_dict={X: feed_vec, X2:ch_val[start:end], S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                            std_val_dnn: std_val, lr: learning_rate, lambda_1_dnn:lam_1_val, lambda_2_dnn:lam_2_val})



                    if i%100 == 0:
                        feed_vec = ch_val[:batch_size]
                        feed_vec = feed_vec.reshape(-1, (num_d2d+1)**2*num_band)
                        feed_diag = ch_val_diag[:batch_size]

                        ######################
                        ## a : Cost  (train)
                        ## b : Rate  (train)
                        ## c : Interference  (train)
                        ## d : EE  (train)
                        ## e: TX power (train)
                        ## f: CUE constraint (train)
                        ## g: DUE constriant (train)
                        ######################
                        a = -sess.run(cost_target,
                                      feed_dict={X: feed_vec, X2:ch_val[:batch_size], S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                 std_val_dnn: std_val, lambda_1_dnn:lam_1_val, lambda_2_dnn:lam_2_val})
                        b = -sess.run(tf_rate,
                                     feed_dict={X: feed_vec, X2:ch_val[:batch_size], S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_1_dnn:lam_1_val})
                        b = b/np.log(2)
                        c = sess.run(CUE_inter_plot,
                                      feed_dict={X: feed_vec, X2:ch_val[:batch_size], S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                 std_val_dnn: std_val, lambda_1_dnn:lam_1_val}) + 1e-15

                        d = -sess.run(tf_ee,
                                     feed_dict={X: feed_vec, X2: ch_val[:batch_size], S_Diag: feed_diag,
                                                p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_1_dnn: lam_1_val})
                        d = d / np.log(2)

                        e = sess.run(py_x_temp,
                                      feed_dict={X: feed_vec, X2: ch_val[:batch_size], S_Diag: feed_diag,
                                                 p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                 std_val_dnn: std_val, lambda_1_dnn: lam_1_val})

                        e = np.sum(e) / batch_size / num_d2d

                        f = sess.run(CUE_inter,
                                     feed_dict={X: feed_vec, X2: ch_val[:batch_size], S_Diag: feed_diag,
                                                p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_1_dnn: lam_1_val})
                        f = np.sum((np.array(f) > 0).astype(float)) / batch_size / num_band


                        g = sess.run(DUE_out,
                                     feed_dict={X: feed_vec, X2: ch_val[:batch_size], S_Diag: feed_diag,
                                                p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_1_dnn: lam_1_val})

                        g = np.sum((np.array(g) > 0).astype(float)) / batch_size / num_d2d


                        feed_vec_1 = ch_val_test[:batch_size]
                        feed_vec_1 = feed_vec_1.reshape(-1, (num_d2d+1)**2*num_band)
                        feed_diag_1 = ch_val_test_diag[:batch_size]

                        ######################
                        ## a_1 : Cost  (test)
                        ## b_1 : Rate  (test)
                        ## c_1 : Interference  (test)
                        ## d_1 : EE  (test)
                        ## e_1 : TX power (test)
                        ## f_1 : CUE constraint (test)
                        ## g_1 : DUE constriant (test)
                        ######################
                        a_1 = -sess.run(cost_target, feed_dict={X: feed_vec_1, X2:ch_val_test[:batch_size], S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                           avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_1_dnn:lam_1_val, lambda_2_dnn:lam_2_val})
                        b_1 = -sess.run(tf_rate, feed_dict={X: feed_vec_1, X2:ch_val_test[:batch_size],S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                           avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_1_dnn:lam_1_val})
                        b_1 = b_1 / np.log(2)
                        c_1 = sess.run(CUE_inter_plot, feed_dict={X: feed_vec_1, X2:ch_val_test[:batch_size],S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                                   avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_1_dnn:lam_1_val}) + 1e-15
                        d_1 = -sess.run(tf_ee,
                                     feed_dict={X: feed_vec_1, X2: ch_val_test[:batch_size], S_Diag: feed_diag_1,
                                                p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_1_dnn: lam_1_val})
                        d_1 = d_1 / np.log(2)

                        e_1 = sess.run(py_x_temp,
                                       feed_dict={X: feed_vec_1, X2: ch_val_test[:batch_size], S_Diag: feed_diag_1,
                                                  p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                  std_val_dnn: std_val, lambda_1_dnn: lam_1_val})
                        e_1 = np.sum(e_1) / batch_size / num_d2d

                        f_1 = sess.run(CUE_inter,
                                       feed_dict={X: feed_vec_1, X2: ch_val_test[:batch_size], S_Diag: feed_diag_1,
                                                  p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                  std_val_dnn: std_val, lambda_1_dnn: lam_1_val})
                        f_1 = np.sum((np.array(f_1) > 0).astype(float)) / batch_size / num_band

                        g_1 = sess.run(DUE_out,
                                       feed_dict={X: feed_vec_1, X2: ch_val_test[:batch_size], S_Diag: feed_diag_1,
                                                  p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                  std_val_dnn: std_val, lambda_1_dnn: lam_1_val})

                        g_1 = np.sum((np.array(g_1) > 0).astype(float)) / batch_size / num_d2d


                        print("")
                        print("Second iter: %d" %i)
                        print("COST: %0.3f, impr = %0.3f in percentage = %0.0f" % (
                            a, (a - a_prev), (a - a_prev) / a * 100))
                        print("RATE: %0.3f, EQ = %0.3f, opt(rate) = %0.3f, opt(ee) = %0.3f, opt(tx) = %0.3f" % (
                            b, ref_eq_rate, ref_opt_rate_rate, ref_opt_ee_rate, ref_opt_tx_rate))
                        print("EE  : %0.3f, EQ = %0.3f, opt(rate) = %0.3f, opt(ee) = %0.3f, opt(tx) = %0.3f" % (
                            d*1e3, ref_eq_ee*1e3, ref_opt_rate_ee*1e3, ref_opt_ee_ee*1e3, ref_opt_tx_ee*1e3))
                        print("TX  : %0.0f, EQ = %0.0f, opt(rate) = %0.0f, opt(ee) = %0.0f, opt(tx) = %0.0f" % (
                            e, ref_eq_tx, ref_opt_rate_tx, ref_opt_ee_tx, ref_opt_tx_tx))
                        print("Inter: %0.3f, EQ = %0.3f, opt(rate) = %0.3f, opt(ee) = %0.3f, opt(tx) = %0.3f" % (
                            10*np.log10(c), 10*np.log10(ref_eq_CUI),
                            10*np.log10(ref_opt_rate_CUI), 10*np.log10(ref_opt_ee_CUI), 10*np.log10(ref_opt_tx_CUI)))
                        print("OUT(CUE): %0.1f, EQ = %0.1f, opt(rate) = %0.1f, opt(ee) = %0.1f, opt(tx) = %0.1f" % (
                            f*1e2, ref_eq_out*1e2, ref_opt_rate_out*1e2, ref_opt_ee_out*1e2, ref_opt_tx_out*1e2))
                        print("OUT(DUE): %0.1f, EQ = %0.1f, opt(rate) = %0.1f, opt(ee) = %0.1f, opt(tx) = %0.1f" % (
                            g*1e2, ref_eq_out_DUE*1e2, ref_opt_rate_out_DUE*1e2, ref_opt_ee_out_DUE*1e2, ref_opt_tx_out_DUE*1e2))


                        print("<Test results> COST= %0.3f, RATE= %0.3f, EE= %0.3f, Inter= %0.3f, "
                              "TX = %0.1f, OUT(CUE) = %0.1f, OUT(DUE) = %0.1f"
                              % (a_1, b_1, d_1*1e3, 10*np.log10(c_1), e_1, f_1*1e2, g_1*1e2))

                        a_prev = a
                        b_prev = b
                        c_prev = c
                        d_prev = d
                        e_prev = e
                        f_prev = f
                        g_prev = g


                        print("**" * 40)


                if reuse == 0:
                    save_path = saver.save(sess, "/tmp/model.ckpt")
                    print("Model saved in path: %s" % save_path)
                else:
                    saver.restore(sess, "/tmp/model.ckpt")
                    print("Model restored.")



                ###############################################
                #### evaluation phase #######################
                ###############################################

                ## For test data set
                time_temp = 0
                time_index = 0          # Hold the number of iteration for test dataset
                diff_pow_val = 0
                cap_infea_num = 0
                int_infea_num = 0
                diff_pow_list = np.zeros(1)

                #################################################################
                ####### Given that the performance of proposed scheme should be examined
                ####### for individual sample, we first take one batch of channel samples
                ####### and make iteration for single batch of samples.
                #################################################################
                for start, end in zip(range(0, len(ch_val_test), batch_size), range(batch_size, len(ch_val_test), batch_size)):

                    #######################################################
                    ### Determining the transmit power of DNN based scheme
                    #######################################################
                    feed_vec = ch_val_test[start:end]
                    feed_vec = feed_vec.reshape(-1, (num_d2d + 1) ** 2 * num_band)
                    feed_diag = ch_val_test_diag[start:end]

                    #######################################################
                    ## pw_temp contains the transmit power of DNN based scheme
                    #######################################################
                    pw_temp = sess.run(py_x_t, feed_dict={X: feed_vec, X2:ch_val_test[start:end], S_Diag: feed_diag, p_keep_conv: 1.0,
                                                        avg_val_dnn: avg_val, std_val_dnn: std_val})

                    ## Call channel vector again since in the upper code, its shape has been changed
                    feed_vec_temp = ch_val_test[start:end]
                    for j in range(batch_size):
                        time_temp = time_temp + 1
                        ###############################################
                        ## Calculate the performance of optimal scheme
                        ###############################################
                        conv_result, feasible_check = sinr_conv_opt_all(p_t, inter_threshold, num_d2d, num_band,
                                                        10 ** (avg_val + std_val * feed_vec_temp[j]), rate_thr)


                        ###############################################
                        ## Calculate the performance only when it is possible to find feasible solution
                        ###############################################
                        if feasible_check == 0:

                            ###################################################
                            ## Calculate the performance of RATE MAXIMIZE
                            ###################################################
                            temp_cap_opt_rate = conv_result[0][0]
                            temp_CUI_opt_rate = conv_result[0][1]
                            temp_OUT_prob_opt_rate = conv_result[0][2]
                            temp_OUT_DUE_opt_rate = conv_result[0][3]
                            temp_tx_pow_opt_rate = conv_result[0][4]

                            cap_opt_rate_DUE = cap_opt_rate_DUE + np.mean(temp_cap_opt_rate[:-1])
                            ee_opt_rate = ee_opt_rate + np.mean(np.divide(temp_cap_opt_rate[:-1], (np.sum(temp_tx_pow_opt_rate, 0)[:-1] + p_c)))
                            out_opt_rate = out_opt_rate + temp_OUT_prob_opt_rate
                            out_DUE_opt_rate = out_DUE_opt_rate + temp_OUT_DUE_opt_rate
                            inter_opt_rate = inter_opt_rate + temp_CUI_opt_rate
                            tx_opt_rate = tx_opt_rate + np.sum(np.sum(temp_tx_pow_opt_rate, 0)[:-1]) / num_d2d


                            ###################################################
                            ## Calculate the performance of EE MAXIMIZE
                            ###################################################
                            temp_cap_opt_ee = conv_result[1][0]
                            temp_CUI_opt_ee = conv_result[1][1]
                            temp_OUT_prob_opt_ee = conv_result[1][2]
                            temp_OUT_DUE_opt_ee = conv_result[1][3]
                            temp_tx_pow_opt_ee = conv_result[1][4]

                            cap_opt_ee_DUE = cap_opt_ee_DUE + np.mean(temp_cap_opt_ee[:-1])
                            ee_opt_ee = ee_opt_ee + np.mean(
                                np.divide(temp_cap_opt_ee[:-1], (np.sum(temp_tx_pow_opt_ee, 0)[:-1] + p_c)))
                            out_opt_ee = out_opt_ee + temp_OUT_prob_opt_ee
                            out_DUE_opt_ee = out_DUE_opt_ee + temp_OUT_DUE_opt_ee
                            inter_opt_ee = inter_opt_ee + temp_CUI_opt_ee
                            tx_opt_ee = tx_opt_ee + np.sum(np.sum(temp_tx_pow_opt_ee, 0)[:-1]) / num_d2d


                            ###################################################
                            ## Calculate the performance of MINIMIZE TX POWER
                            ###################################################
                            temp_cap_opt_tx = conv_result[2][0]
                            temp_CUI_opt_tx = conv_result[2][1]
                            temp_OUT_prob_opt_tx = conv_result[2][2]
                            temp_OUT_DUE_opt_tx = conv_result[2][3]
                            temp_tx_pow_opt_tx = conv_result[2][4]

                            cap_opt_tx_DUE = cap_opt_tx_DUE + np.mean(temp_cap_opt_tx[:-1])
                            ee_opt_tx = ee_opt_tx + np.mean(
                                np.divide(temp_cap_opt_tx[:-1], (np.sum(temp_tx_pow_opt_tx, 0)[:-1] + p_c)))
                            out_opt_tx = out_opt_tx + temp_OUT_prob_opt_tx
                            out_DUE_opt_tx = out_DUE_opt_tx + temp_OUT_DUE_opt_tx
                            inter_opt_tx = inter_opt_tx + temp_CUI_opt_tx
                            tx_opt_tx = tx_opt_tx + np.sum(np.sum(temp_tx_pow_opt_tx, 0)[:-1]) / num_d2d



                            ##########################################
                            ## Determine transmit power for DNN  (Test set)
                            ##########################################
                            temp_cap_dnn_val, temp_inter_dnn, temp_OUT_prob_dnn, temp_OUT_DUE_dnn, _ = sinr_dnn_infeasible(pw_temp[j],
                                                                                                                inter_threshold, num_d2d,
                                                                                                                num_band, 10 ** (avg_val + std_val * feed_vec_temp[j]),
                                                                                                                rate_thr)




                            temp_cap_infea = np.array(temp_cap_dnn_val[:-1])
                            temp_cap_infea_bol = np.sum((temp_cap_infea < rate_thr).astype("float"))
                            if temp_cap_infea_bol > 0:
                                temp_cap_infea = np.sum(temp_cap_infea[temp_cap_infea<rate_thr])
                            else:
                                temp_cap_infea = 0
                            cap_infea_num = cap_infea_num + temp_cap_infea_bol


                            temp_int_infea = np.array(temp_inter_dnn)
                            temp_int_infea_bol = np.sum((temp_int_infea > inter_threshold).astype("float"))
                            if temp_int_infea_bol > 0:
                                temp_int_infea = np.sum(temp_int_infea[temp_int_infea > inter_threshold])
                            else:
                                temp_int_infea = 0
                            int_infea_num = int_infea_num + temp_int_infea_bol



                            cap_te_DUE = cap_te_DUE + temp_cap_infea
                            cap_te_CUE = cap_te_CUE + np.mean(temp_cap_dnn_val[-1:])/num_band
                            ee_te = ee_te + np.mean(
                                np.divide(temp_cap_dnn_val[:-1], (np.sum(pw_temp[j], 0)[:-1] + p_c)))
                            out_te = out_te + temp_OUT_prob_dnn
                            out_DUE_te = out_DUE_te + temp_OUT_DUE_dnn
                            inter_te = inter_te + temp_int_infea
                            tx_te = tx_te + np.sum(np.sum(pw_temp[j], 0)[:-1]) / num_d2d

                            print("temp_cap_infea (cur) = ", temp_cap_infea)
                            print("temp_int (cur) = ", np.log10(temp_int_infea))
                            print("")
                            print("temp_cap_infea (tot) = ", cap_te_DUE/cap_infea_num)
                            print("temp_int (tot) = ", np.log10(inter_te/int_infea_num))
                            print("")
                            print("cap_infea_num = ", cap_infea_num)
                            print("int_infea_num = ", int_infea_num)
                            print("")
                            print("")
                            print("")







                            ###############################
                            ## Determine transmit power for equal power scheme
                            ################################
                            temp_cap_eq, temp_inter_eq, temp_OUT_prob_eq, temp_OUT_prob_eq, temp_tx_pow_eq = sinr_eq(p_t, inter_threshold, num_d2d, num_band,
                                                                  10 ** (avg_val + std_val * feed_vec_temp[j]), rate_thr)

                            cap_eq_DUE = cap_eq_DUE + np.mean(temp_cap_eq[:-1])
                            cap_eq_CUE = cap_eq_CUE + np.mean(temp_cap_eq[-1:])/num_band
                            ee_eq = ee_eq + np.mean(
                                np.divide(temp_cap_eq[:-1], (np.sum(temp_tx_pow_eq, 0)[:-1] + p_c)))
                            out_eq = out_eq + temp_OUT_prob_eq
                            inter_eq = inter_eq + temp_inter_eq
                            tx_eq = tx_eq + np.sum(temp_tx_pow_eq[:-1]) / num_d2d

                            if target == 0:
                                diff_pow = pw_temp[j][:, :-1] - temp_tx_pow_opt_rate[:, :-1]
                            elif target == 1:
                                diff_pow = pw_temp[j][:, :-1] - temp_tx_pow_opt_ee[:, :-1]
                            else:
                                diff_pow = pw_temp[j][:, :-1] - temp_tx_pow_opt_tx[:, :-1]


                            diff_pow_list = np.append(diff_pow_list, diff_pow.reshape([-1]))
                            diff_pow_val = diff_pow_val + LA.norm(diff_pow)


                            if time_index%20 == 0:
                                print(time_index)
                                print("DNN pow = ", pw_temp[j][:, :-1] )
                                print("Opt pow = ", temp_tx_pow_opt_rate[:, :-1])
                                print("diff_pow (cur) = ", diff_pow)
                                print("diff_pow (tot) = ", diff_pow_val/(time_index+1)/4)
                                print("Rate temp = ", cap_opt_rate_DUE / (time_index + 1))
                                print("CUrrent rate = ", np.mean(temp_cap_opt_rate[:-1]))

                            time_index = time_index + 1





                np.savetxt("diff_pow.csv", diff_pow_list, delimiter=",")


                ## Normalize the performance of DNN and Equal power
                cap_te_DUE, cap_eq_DUE = cap_te_DUE / cap_infea_num, cap_eq_DUE / time_index
                cap_te_CUE, cap_eq_CUE = cap_te_CUE / time_index, cap_eq_CUE / time_index
                out_te, out_eq = out_te / time_index, out_eq / time_index
                out_DUE_te, out_DUE_eq = out_DUE_te / time_index, out_DUE_eq / time_index
                inter_te, inter_eq = inter_te / int_infea_num, inter_eq / time_index
                ee_te, ee_eq = ee_te / time_index, ee_eq / time_index
                tx_te, tx_eq = tx_te / time_index, tx_eq / time_index

                cap_opt_rate_DUE, cap_opt_ee_DUE, cap_opt_tx_DUE = cap_opt_rate_DUE / time_index, cap_opt_ee_DUE / time_index, cap_opt_tx_DUE / time_index
                cap_opt_rate_CUE, cap_opt_ee_CUE, cap_opt_tx_CUE = cap_opt_rate_CUE / time_index, cap_opt_ee_CUE / time_index, cap_opt_tx_CUE / time_index
                out_opt_rate, out_opt_ee, out_opt_tx  = out_opt_rate / time_index, out_opt_ee / time_index, out_opt_tx / time_index
                out_DUE_opt_rate, out_DUE_opt_ee, out_DUE_opt_tx = out_DUE_opt_rate / time_index, out_DUE_opt_ee / time_index, out_DUE_opt_tx / time_index
                inter_opt_rate, inter_opt_ee, inter_opt_tx = inter_opt_rate / time_index, inter_opt_ee / time_index, inter_opt_tx / time_index
                ee_opt_rate, ee_opt_ee, ee_opt_tx = ee_opt_rate/time_index, ee_opt_ee/time_index, ee_opt_tx/time_index
                tx_opt_rate, tx_opt_ee, tx_opt_tx = tx_opt_rate / time_index, tx_opt_ee / time_index, tx_opt_tx / time_index


                #####################
                ## 2 is training and 1 is test
                print("")
                print("")
                print("Rate (DUE): DNN => %0.2f, EQ => %0.2f, OPT(RATE) => %0.2f, OPT(EE) => %0.2f, OPT(TX) => %0.2f" % (
                    cap_te_DUE, cap_eq_DUE, cap_opt_rate_DUE, cap_opt_ee_DUE, cap_opt_tx_DUE))
                print("EE (DUE): DNN(test) => %0.1f, EQ => %0.1f, OPT(RATE) => %0.1f, OPT(EE) => %0.1f, OPT(TX) => %0.1f" % (
                    ee_te*1e3, ee_eq*1e3, ee_opt_rate*1e3, ee_opt_ee*1e3, ee_opt_tx*1e3))
                print("Out(CUE): DNN(test) => %0.1f, EQ => %0.1f, OPT(RATE) => %0.1f, OPT(EE) => %0.1f, OPT(TX) => %0.1f" % (
                    out_te*1e2, out_eq*1e2, out_opt_rate*1e2, out_opt_ee*1e2, out_opt_tx*1e2))
                print("Out(DUE): DNN(test) => %0.1f, EQ => %0.1f, OPT(RATE) => %0.1f, OPT(EE) => %0.1f, OPT(TX) => %0.1f" % (
                    out_DUE_te*1e2, out_DUE_eq*1e2, out_DUE_opt_rate*1e2, out_DUE_opt_ee*1e2, out_DUE_opt_tx*1e2))
                print("Inter: DNN(test) => %0.1f, EQ => %0.1f, OPT(RATE) => %0.1f, OPT(EE) => %0.1f, OPT(TX) => %0.1f" % (
                    10*math.log10(inter_te), 10*math.log10(inter_eq), 10*math.log10(inter_opt_rate), 10*math.log10(inter_opt_ee), 10*math.log10(inter_opt_tx)))
                print("TX power (DUE): DNN => %0.0f, EQ => %0.0f, OPT(RATE) => %0.0f, OPT(EE) => %0.0f, OPT(TX) => %0.0f" % (
                    tx_te, tx_eq, tx_opt_rate, tx_opt_ee, tx_opt_tx))
                print("Rate (CUE): DNN(test) => %0.3f, EQ => %0.3f, Inter => %0.3f, OPT(RATE) => %0.3f, OPT(EE) => %0.3f, OPT(TX) => %0.3f" % (cap_te_CUE, cap_eq_CUE, cap_inter_CUE, cap_opt_rate_CUE, cap_opt_ee_CUE, cap_opt_tx_CUE))





            ###########################
            ## Save to matrix #########
            ###########################
            ###### DNN - Test set #####
            ###########################
            cap_mat_te_DUE[k, :] = cap_mat_te_DUE[k, :] + cap_te_DUE
            cap_mat_te_CUE[k, :] = cap_mat_te_CUE[k, :] + cap_te_CUE
            ee_mat_te[k, :] = ee_mat_te[k, :] + ee_te
            out_mat_te[k, :] = out_mat_te[k, :] + out_te
            out_DUE_mat_te[k, :] = out_DUE_mat_te[k, :] + out_DUE_te
            inter_mat_te[k, :] = inter_mat_te[k, :] + inter_te
            tx_mat_te[k, :] = tx_mat_te[k, :] + tx_te

            ###########################
            ###### Equal power #######
            ###########################
            cap_mat_eq_DUE[k, :] = cap_mat_eq_DUE[k, :] + cap_eq_DUE
            cap_mat_eq_CUE[k, :] = cap_mat_eq_CUE[k, :] + cap_eq_CUE
            ee_mat_eq[k, :] = ee_mat_eq[k, :] + ee_eq
            out_mat_eq[k, :] = out_mat_eq[k, :] + out_eq
            out_DUE_mat_eq[k, :] = out_DUE_mat_eq[k, :] + out_DUE_eq
            inter_mat_eq[k, :] = inter_mat_eq[k, :] + inter_eq
            tx_mat_eq[k, :] = tx_mat_eq[k, :] + tx_eq

            ##################################
            ###### Optimal scheme (RATE) #####
            ##################################
            cap_mat_opt_rate_DUE[k, :] = cap_mat_opt_rate_DUE[k, :] + cap_opt_rate_DUE
            cap_mat_opt_rate_CUE[k, :] = cap_mat_opt_rate_CUE[k, :] + cap_opt_rate_CUE
            ee_mat_opt_rate[k, :] = ee_mat_opt_rate[k, :] + ee_opt_rate
            out_mat_opt_rate[k, :] = out_mat_opt_rate[k, :] + out_opt_rate
            out_DUE_mat_opt_rate[k, :] = out_DUE_mat_opt_rate[k, :] + out_DUE_opt_rate
            inter_mat_opt_rate[k, :] = inter_mat_opt_rate[k, :] + inter_opt_rate
            tx_mat_opt_rate[k, :] = tx_mat_opt_rate[k, :] + tx_opt_rate

            ##################################
            ###### Optimal scheme (EE) #####
            ##################################
            cap_mat_opt_ee_DUE[k, :] = cap_mat_opt_ee_DUE[k, :] + cap_opt_ee_DUE
            cap_mat_opt_ee_CUE[k, :] = cap_mat_opt_ee_CUE[k, :] + cap_opt_ee_CUE
            ee_mat_opt_ee[k, :] = ee_mat_opt_ee[k, :] + ee_opt_ee
            out_mat_opt_ee[k, :] = out_mat_opt_ee[k, :] + out_opt_ee
            out_DUE_mat_opt_ee[k, :] = out_DUE_mat_opt_ee[k, :] + out_DUE_opt_ee
            inter_mat_opt_ee[k, :] = inter_mat_opt_ee[k, :] + inter_opt_ee
            tx_mat_opt_ee[k, :] = tx_mat_opt_ee[k, :] + tx_opt_ee


            ##################################
            ###### Optimal scheme (TX) #####
            ##################################
            cap_mat_opt_tx_DUE[k, :] = cap_mat_opt_tx_DUE[k, :] + cap_opt_tx_DUE
            cap_mat_opt_tx_CUE[k, :] = cap_mat_opt_tx_CUE[k, :] + cap_opt_tx_CUE
            ee_mat_opt_tx[k, :] = ee_mat_opt_tx[k, :] + ee_opt_tx
            out_mat_opt_tx[k, :] = out_mat_opt_tx[k, :] + out_opt_tx
            out_DUE_mat_opt_tx[k, :] = out_DUE_mat_opt_tx[k, :] + out_DUE_opt_tx
            inter_mat_opt_tx[k, :] = inter_mat_opt_tx[k, :] + inter_opt_tx
            tx_mat_opt_tx[k, :] = tx_mat_opt_tx[k, :] + tx_opt_tx


            ###############################
            ## Print final results
            ###############################



            ###  Print Rate - DUE
            print("**" * 40)
            print("   " * 40)
            print("Rate(DUE): DNN (Test) = ", np.transpose(cap_mat_te_DUE/iter_num))
            print("Rate(DUE): OPT (RATE)  = ", np.transpose(cap_mat_opt_rate_DUE / iter_num))
            print("Rate(DUE): OPT (EE)  = ", np.transpose(cap_mat_opt_ee_DUE / iter_num))
            print("Rate(DUE): OPT (TX)  = ", np.transpose(cap_mat_opt_tx_DUE / iter_num))
            print("Rate(DUE): EQ  = ", np.transpose(cap_mat_eq_DUE/iter_num))
            print("**" * 40)
            print("   " * 40)


            ###  Print - EE
            print("**" * 40)
            print("   " * 40)
            print("EE: DNN (Test) = ", 1e3*np.transpose(ee_mat_te/iter_num))
            print("EE: OPT (RATE)  = ", 1e3*np.transpose(ee_mat_opt_rate / iter_num))
            print("EE: OPT (EE)  = ", 1e3*np.transpose(ee_mat_opt_ee / iter_num))
            print("EE: OPT (TX)  = ", 1e3*np.transpose(ee_mat_opt_tx / iter_num))
            print("EE: EQ  = ", 1e3*np.transpose(ee_mat_eq/iter_num))
            print("**" * 40)
            print("   " * 40)


            ###  Print - TX_power
            print("**" * 40)
            print("   " * 40)
            print("TX: DNN (Test) = ", np.transpose(tx_mat_te/iter_num))
            print("TX: OPT (RATE)  = ", np.transpose(tx_mat_opt_rate / iter_num))
            print("TX: OPT (EE)  = ", np.transpose(tx_mat_opt_ee / iter_num))
            print("TX: OPT (TX)  = ", np.transpose(tx_mat_opt_tx / iter_num))
            print("TX: EQ  = ", np.transpose(tx_mat_eq/iter_num))
            print("**" * 40)
            print("   " * 40)


            ###  Print - OUT - CUE
            print("**" * 40)
            print("   " * 40)
            print("OUT(CUE): DNN (Test) = ", np.transpose(out_mat_te/iter_num))
            print("OUT(CUE): OPT (RATE)  = ", np.transpose(out_mat_opt_rate / iter_num))
            print("OUT(CUE): OPT (EE)  = ", np.transpose(out_mat_opt_ee / iter_num))
            print("OUT(CUE): OPT (TX)  = ", np.transpose(out_mat_opt_tx / iter_num))
            print("OUT(CUE): EQ  = ", np.transpose(out_mat_eq/iter_num))
            print("**" * 40)
            print("   " * 40)


            ###  Print - OUT - DUE
            print("**" * 40)
            print("   " * 40)
            print("OUT(DUE): DNN (Test) = ", np.transpose(out_DUE_mat_te/iter_num))
            print("OUT(DUE): OPT (RATE)  = ", np.transpose(out_DUE_mat_opt_rate / iter_num))
            print("OUT(DUE): OPT (EE)  = ", np.transpose(out_DUE_mat_opt_ee / iter_num))
            print("OUT(DUE): OPT (TX)  = ", np.transpose(out_DUE_mat_opt_tx / iter_num))
            print("OUT(DUE): EQ  = ", np.transpose(out_DUE_mat_eq/iter_num))
            print("**" * 40)
            print("   " * 40)


            ###  Print - Interference
            print("**" * 40)
            print("   " * 40)
            print("Inter: DNN (Test) = ", np.transpose(10*np.log10(inter_mat_te/iter_num)))
            print("Inter: OPT (RATE)  = ", np.transpose(10 * np.log10(inter_mat_opt_rate / iter_num)))
            print("Inter: OPT (EE)  = ", np.transpose(10 * np.log10(inter_mat_opt_ee / iter_num)))
            print("Inter: OPT (TX)  = ", np.transpose(10 * np.log10(inter_mat_opt_tx / iter_num)))
            print("Inter: EQ  = ", np.transpose(10 * np.log10(inter_mat_eq / iter_num)))
            print("**" * 40)
            print("   " * 40)

            ###  Print Rate - CUE
            print("**" * 40)
            print("   " * 40)
            print("Rate(CUE): DNN(Test) = ", np.transpose(cap_mat_te_CUE/iter_num))
            print("Rate(CUE): OPT (RATE)  = ", np.transpose(cap_mat_opt_rate_CUE / iter_num))
            print("Rate(CUE): OPT (EE)  = ", np.transpose(cap_mat_opt_ee_CUE / iter_num))
            print("Rate(CUE): OPT (TX)  = ", np.transpose(cap_mat_opt_tx_CUE / iter_num))
            print("Rate(CUE): EQ  = ", np.transpose(cap_mat_eq_CUE/iter_num))
            print("**" * 40)
            print("   " * 40)

    return 0



'''
    Channel related parameters
'''
#########################################################################
###############       DO NOT CHANGE            #########################
#########################################################################
# size of bandwith - 10MHz
bw = 10*10**6
p_t_dB = 23.
p_t = 10**(p_t_dB/10)
p_c_dB = 20.        ### Should be change
p_c = 10**(p_c_dB/10)
pl_const = 34.5
pl_alpha = 38.
pl_const_test = 34.5
pl_alpha_test = 38.0
N0W = bw*10**(-174./10)   # Noise: -174 dBm/Hz
avg_val = 1.
std_val = 1.
# size of area sensors are distributed = 500
d2d_dist = 15.
#########################################################################
###############       Change            #########################
#########################################################################
#########################################################################
###############       Change            #########################
#########################################################################
#########################################################################
###############       Change            #########################
#########################################################################
#########################################################################
###############       Change            #########################
#########################################################################


delta_val = N0W*1e6
delta_val_tx_pow = 2.0/2
iter_num = 1
hidden_num = 100
batch_size = 100
num_d2d = 2
num_band = 2
tot_epoch_real = 2000


# generated channel sample
num_samples = 40000
inter_threshold = 10**(-55.0/10)
rate_thr = 3

inter_threshold_mar = inter_threshold
rate_thr_mar = rate_thr


test_size = np.minimum(30*batch_size+1, num_samples)
## if reuse == 0, redo trainig, 1-> intefence
learning_rate_init = 1e-4
reuse = 0
granu = 15
target = 0  # 0: rate max, 1: EE max, 2: tx pow_max
gamma_1 = 2.0
gamma_2 = 1.0


#########################################################################
###############       Change            #########################
#########################################################################
#########################################################################
###############       Change            #########################
#########################################################################
#########################################################################
###############       Change            #########################
#########################################################################
#########################################################################
###############       Change            #########################
#########################################################################





'''
    DNN related parameters
'''

# X is the channel information including DUE-DUE channel and DUE-CUE channel
X = tf.placeholder(tf.float32, [batch_size, num_band*(num_d2d+1)**2])

# X2 is the reshaped value of X which fascilate the operation
X2 = tf.placeholder(tf.float32, [batch_size, num_band, num_d2d+1, num_d2d+1])

# S_Diag contains the signal_channel value which fascilate the operation
S_Diag = tf.placeholder(tf.float32, [batch_size, num_band, num_d2d+1])

p_keep_conv = tf.placeholder(tf.float32)
avg_val_dnn = tf.placeholder(tf.float64)
std_val_dnn = tf.placeholder(tf.float64)
lambda_1_dnn = tf.placeholder(tf.float64)
lambda_2_dnn = tf.placeholder(tf.float64)

lr = tf.placeholder("float")


stddev_init = tf.sqrt(3.0 / (hidden_num + hidden_num))

## Define variables. The bias for conv layer is defined at the model function.
w1 = tf.Variable(tf.random_normal((num_band*(num_d2d+1)**2, hidden_num), stddev=stddev_init))
w2 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
w3 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
w4 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
w5 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
wo = tf.Variable(tf.random_normal((hidden_num, num_band*(num_d2d)), stddev=stddev_init))
wp = tf.Variable(tf.random_normal((hidden_num, num_d2d), stddev=stddev_init))


w1_1 = tf.Variable(tf.random_normal((num_band*(num_d2d+1)**2, hidden_num), stddev=stddev_init))
w2_1 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
w3_1 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
w4_1 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))
w5_1 = tf.Variable(tf.random_normal((hidden_num, hidden_num), stddev=stddev_init))


b1 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b2 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b3 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b4 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b5 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
bo = tf.Variable(tf.random_normal((1, num_band*(num_d2d)), stddev=stddev_init))
bp = tf.Variable(tf.random_normal((1, num_d2d), stddev=stddev_init))

b1_1 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b2_1 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b3_1 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b4_1 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))
b5_1 = tf.Variable(tf.random_normal((1, hidden_num), stddev=stddev_init))


py_x = p_t * model(X, w1, w2, w3, w4, w5, w1_1, w2_1, w3_1, w4_1, w5_1, wo, wp, b1, b2, b3, b4, b5, b1_1, b2_1, b3_1, b4_1, b5_1,bo, bp, p_keep_conv, num_d2d, num_band)
py_x_temp = tf.reshape(py_x, [batch_size, num_d2d, num_band])

## pt_cell models the transmit power of CUE
pt_cell = p_t*tf.ones([py_x.get_shape().as_list()[0], 1, num_band], tf.float32)

## py_x_t is the concanated transmit power of DUE and CUE
py_x_t = tf.cast(tf.transpose(tf.concat([py_x_temp, pt_cell], 1), perm=[0, 2, 1]), dtype=tf.float64)




ch_temp = tf.transpose(10**(std_val_dnn*tf.cast(X2, dtype=tf.float64)+avg_val_dnn), perm=[0, 1, 3, 2])

sig_pw = tf.multiply(py_x_t, 10**(std_val_dnn*tf.cast(S_Diag, dtype=tf.float64)+avg_val_dnn))
int_pw_1 = tf.multiply(tf.reshape(py_x_t, [-1, num_band, num_d2d+1, 1]), ch_temp)
int_pw_2 = tf.reshape(tf.reduce_sum(int_pw_1, 2), [-1, num_band, num_d2d+1])

SINR = tf.div(sig_pw, int_pw_2 - sig_pw + N0W)
cap_val = tf.log(1 + SINR)



### CUE_inter contains the difference between interference and threshold
CUE_inter = tf.nn.relu((int_pw_2 - sig_pw)[:,:,-1:] - tf.constant(inter_threshold_mar, dtype=tf.float64))
CUE_inter_plot = tf.reduce_mean((int_pw_2 - sig_pw)[:,:,-1:])

## Rate is negative value
tf_rate = tf.reduce_mean(tf.reduce_sum(-cap_val[:,:,:-1], 1))
tf_ee = tf.reduce_mean(tf.div(tf.reduce_sum(-cap_val[:,:,:-1], 1), tf.reduce_sum(py_x_t[:,:,:-1], 1)+p_c))

### CUE_inter contains the difference between interference and threshold
DUE_out = tf.nn.relu(tf.constant(rate_thr_mar*np.log(2), dtype=tf.float64) - tf.reduce_sum(cap_val[:,:,:-1], 1))

### Sum tx power
dnn_sum_tx_power = tf.cast(tf.reduce_mean(tf.reduce_sum(py_x_temp, 1)), dtype=tf.float64)

cost_rate = tf_rate + lambda_1_dnn*tf.reduce_mean(tf.nn.tanh(CUE_inter/delta_val))+ lambda_2_dnn*tf.reduce_mean(tf.nn.tanh(DUE_out/delta_val_tx_pow))
cost_ee = tf_ee + lambda_1_dnn*tf.reduce_mean(tf.nn.tanh(CUE_inter/delta_val))+ lambda_2_dnn*tf.reduce_mean(tf.nn.tanh(DUE_out/delta_val_tx_pow))
cost_tx = dnn_sum_tx_power + lambda_1_dnn*tf.reduce_mean(tf.nn.tanh(CUE_inter/delta_val))+ lambda_2_dnn*tf.reduce_mean(tf.nn.tanh(DUE_out/delta_val_tx_pow))

temp_val_dnn  = lambda_2_dnn*tf.reduce_mean(tf.nn.tanh(DUE_out/delta_val_tx_pow))

train_op_rate = tf.train.AdamOptimizer(lr).minimize(cost_rate)
train_op_ee = tf.train.AdamOptimizer(lr).minimize(cost_ee)
train_op_tx = tf.train.AdamOptimizer(lr).minimize(cost_tx)




#cost_1 = (1.0-lambda_dnn)*tf_rate + lambda_dnn*tf.reduce_mean(tf.nn.tanh(CUE_inter/delta_val))
#cost_1 = (1.0-lam_val)*tf_rate + lam_val*tf.reduce_mean(tf.nn.tanh(CUE_inter/N0W))
#cost_1 = (1.0-lam_val)*tf_rate + lam_val*CUE_inter_prob
#cost_1 = tf_rate


######################################################################################33

per_eval(batch_size, inter_threshold, num_band, rate_thr, learning_rate_init, target)

