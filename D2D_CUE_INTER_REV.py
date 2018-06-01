import tensorflow as tf
import numpy as np
import math
import os
import time

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#np.set_printoptions(threshold=np.nan)


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

    VV = np.zeros(10)
    for iter in range(10):
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

'''

## Calculate SINR of eq given channel pt and num d2d
def sinr_eq(p_t, inter_threshold, num_d2d, ch_val):
    tx_pow = p_t * np.ones((num_d2d+1, 1))
    ch_val_1 = np.array(ch_val, copy=True)
    ch_w_eq = np.multiply(tx_pow, ch_val_1)
    sig_eq = np.array(np.diag(ch_w_eq), copy=True)
    int_eq = np.sum(np.transpose(ch_w_eq), 1)
    out_prob = np.mean((int_eq - sig_eq)[-1:]> inter_threshold)
    return np.divide(sig_eq, int_eq - sig_eq + N0W), out_prob
'''


## Calculate SINR of eq given channel pt and num d2d
def sinr_eq(p_t, inter_threshold, num_d2d, ch_val):

    ## Calculating the proper tx power for eq scheme
    tx_pow = np.ones((num_d2d+1, 1))
    ch_val_1 = np.array(ch_val, copy=True)
    ch_w_eq = np.multiply(tx_pow, ch_val_1)
    sig_eq = np.array(np.diag(ch_w_eq), copy=True)
    int_eq = np.sum(np.transpose(ch_w_eq), 1)
    out_val = inter_threshold/np.mean((int_eq - sig_eq)[-1:])
    prop_t = np.minimum(p_t, np.maximum(out_val-1e-5, 0.0))

    tx_pow = prop_t*np.ones((num_d2d + 1, 1))
    tx_pow[-1:] = p_t
    ch_val_1 = np.array(ch_val, copy=True)
    ch_w_eq = np.multiply(tx_pow, ch_val_1)
    sig_eq = np.array(np.diag(ch_w_eq), copy=True)
    int_eq = np.sum(np.transpose(ch_w_eq), 1)
    out_prob = np.mean((int_eq - sig_eq)[-1:]> inter_threshold)

    return np.divide(sig_eq, int_eq - sig_eq + N0W), out_prob, (int_eq - sig_eq)[-1:]



## Calculate SINR of eq given channel pt and num d2d
def sinr_wm(p_t, inter_threshold, num_d2d, ch_val):

    ## calculating interference from cellular user
    ## int_cell contains the interference caused by the CUE
    tx_pow_cell = p_t * np.ones((num_d2d + 1, 1))
    tx_pow_cell[:-1,] = 0
    ch_cell = np.multiply(tx_pow_cell, ch_val)
    int_cell = np.sum(np.transpose(ch_cell), 1) + N0W

    ## Determine the WMMSE output
    ## Note that the WMMSE output's shape is [num_d2d, 1]
    tx_pow = p_t * np.ones((num_d2d, 1))
    tx_pow_wm = WMMSE_sum_rate(tx_pow, np.sqrt(ch_val[:-1,:-1]), p_t, int_cell)

    ## TX power of CUE should be added
    tx_pow_wm = np.vstack((tx_pow_wm,p_t))

    ch_w_wm = np.multiply(tx_pow_wm, ch_val)
    sig_wm = np.array(np.diag(ch_w_wm))
    int_wm = np.sum(np.transpose(ch_w_wm), 1)
    out_prob = np.mean((int_wm - sig_wm)[-1:]> inter_threshold)
    return np.divide(sig_wm, int_wm - sig_wm + N0W), tx_pow_wm, out_prob, (int_wm - sig_wm)[-1:]




## Calculate SINR of eq given channel pt and num d2d
def sinr_dnn(tx_pow, inter_threshold, num_d2d, ch_val):
    tx_pow = np.reshape(tx_pow, [num_d2d+1, 1])
    ch_w_dnn = np.multiply(tx_pow, ch_val)
    sig_dnn = np.array(np.diag(ch_w_dnn))
    int_dnn = np.sum(np.transpose(ch_w_dnn), 1)
    out_prob = np.mean((int_dnn - sig_dnn)[-1:]> inter_threshold)
    return np.divide(sig_dnn, int_dnn- sig_dnn + N0W), out_prob, (int_dnn - sig_dnn)[-1:]


'''
    Building DNN model
'''
def model(X, w1, w2, w3, w4, w5, w6, w7, wo, b1, bo, p_keep_conv, num_d2d):

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    l1_1 = tf.reshape(l1, [-1, num_d2d+1, num_d2d+1, 1])


    l2 = tf.nn.conv2d(l1_1, w2, strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.relu(l2 + tf.Variable(tf.random_normal([l2.get_shape().as_list()[-1]])))
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME')
    l3 = tf.nn.relu(l3 + tf.Variable(tf.random_normal([l3.get_shape().as_list()[-1]])))
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='SAME')
    l4 = tf.nn.relu(l4 + tf.Variable(tf.random_normal([l4.get_shape().as_list()[-1]])))
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5 = tf.nn.conv2d(l4, w5, strides=[1, 1, 1, 1], padding='SAME')
    l5 = tf.nn.relu(l5 + tf.Variable(tf.random_normal([l5.get_shape().as_list()[-1]])))
    l5 = tf.nn.dropout(l5, p_keep_conv)

    l6 = tf.nn.conv2d(l5, w6, strides=[1, 1, 1, 1], padding='SAME')
    l6 = tf.nn.relu(l6 + tf.Variable(tf.random_normal([l6.get_shape().as_list()[-1]])))
    l6 = tf.nn.dropout(l6, p_keep_conv)

    l7 = tf.nn.conv2d(l6, w7, strides=[1, 1, 1, 1], padding='SAME')
    l7 = tf.nn.relu(l7 + tf.Variable(tf.random_normal([l7.get_shape().as_list()[-1]])))

    l7 = tf.reshape(l7, [-1, wo.get_shape().as_list()[0]])

    pyx = tf.nn.sigmoid(tf.matmul(l7, wo) + bo)

    return pyx



'''
    Initialization of location information
'''
def loc_init(size_area, d2d_dist, num_d2d):
    rx_loc = size_area * (np.random.rand(num_d2d + 1, 2) - 0.5)
    rx_loc[num_d2d, :] = np.zeros((2,))
    tx_loc = np.zeros((num_d2d + 1, 2))
    for i in range(num_d2d):
        temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
        temp_chan = rx_loc[i, :] + temp_dist
        while (np.max(abs(temp_chan)) > size_area / 2) | (np.linalg.norm(temp_dist) > d2d_dist):
            temp_dist = d2d_dist * (np.random.rand(1, 2) - 0.5)
            temp_chan = rx_loc[i, :] + temp_dist
        tx_loc[i, :] = temp_chan

    ## Determining the location of CUE
    temp_chan = size_area * (np.random.rand(1, 2) - 0.5)
    while np.max(abs(temp_chan)) > size_area / 2:
        temp_chan = size_area * (np.random.rand(1, 2) - 0.5)
    tx_loc[num_d2d, :] = temp_chan


    return rx_loc, tx_loc



'''
    For the returned matrix, pu_ch_gain[0, : ] indicates the channel of RX 1
'''


def ch_gen(size_area, d2d_dist, num_d2d, num_samples, p_t):
    ch_w_fading = []
    ch_wo_fading = []

    for i in range(num_samples):
        rx_loc, tx_loc = loc_init(size_area, d2d_dist, num_d2d)

        ## generate distance_vector
        dist_vec = rx_loc.reshape(num_d2d+1, 1, 2) - tx_loc
        dist_vec = np.linalg.norm(dist_vec, axis=2)
        dist_vec = np.maximum(dist_vec, 3)

        # find path loss // shadowing is not considered
        pu_ch_gain_db = - pl_const - pl_alpha * np.log10(dist_vec)
        pu_ch_gain = 10 ** (pu_ch_gain_db / 10)
        multi_fading = 0.5 * np.random.randn(num_d2d+1, num_d2d+1) ** 2 + 0.5 * np.random.randn(num_d2d+1, num_d2d+1) ** 2

        final_ch = np.maximum(pu_ch_gain *multi_fading, np.exp(-30))
        ch_w_fading.append(final_ch)
        ch_wo_fading.append(pu_ch_gain)

    return np.array(ch_w_fading), np.array(ch_wo_fading)




'''
    For the returned matrix, pu_ch_gain[0, : ] indicates the channel of RX 1
'''

def per_eval(batch_size, inter_threshold, metric=0, perm_val=0):
    cap_mat_1_DUE = np.zeros((5, 1))
    cap_mat_1_CUE = np.zeros((5, 1))
    cap_mat_2_DUE = np.zeros((5, 1))
    cap_mat_2_CUE = np.zeros((5, 1))

    cap_mat_eq_DUE = np.zeros((5, 1))
    cap_mat_eq_CUE = np.zeros((5, 1))
    cap_mat_wm_DUE = np.zeros((5, 1))
    cap_mat_wm_CUE = np.zeros((5, 1))

    out_mat_1 = np.zeros((5, 1))
    out_mat_2 = np.zeros((5, 1))
    out_mat_eq = np.zeros((5, 1))
    out_mat_wm = np.zeros((5, 1))

    Inter_mat_1 = np.zeros((5, 1))
    Inter_mat_2 = np.zeros((5, 1))
    Inter_mat_eq = np.zeros((5, 1))
    Inter_mat_wm = np.zeros((5, 1))

    lam_val_mat = [0, 0.9, 0.99, 0.999, 0.9995]

    for k in range(5):
        print("iteration: ", k)
        size_area = 200
        lam_val = lam_val_mat[k]
        for l in range(iter_num):
            ch_val_tot, pl_val_tot = ch_gen(size_area, d2d_dist, num_d2d, num_samples, p_t)
            cap_1_DUE, cap_2_DUE, cap_eq_DUE, cap_wm_DUE = 0, 0, 0, 0
            cap_1_CUE, cap_2_CUE, cap_eq_CUE, cap_wm_CUE = 0, 0, 0, 0
            ee_1, ee_2, ee_eq, ee_wm = 0, 0, 0, 0
            out_1, out_2, out_eq, out_wm = 0, 0, 0, 0
            Inter_1, Inter_2, Inter_eq, Inter_wm = 0, 0, 0, 0
            power_1, power_2, power_3 = 0, 0, 0

            _sample_ch = np.log10(ch_val_tot)
            avg_val = np.mean(_sample_ch)
            std_val = np.sqrt(np.var(_sample_ch))

            _sample_ch = (_sample_ch - avg_val) / std_val

            sample_ch = np.array(_sample_ch, copy=True)

            ch_val = sample_ch[:int(0.5*num_samples)]
            ch_val_test = sample_ch[int(0.5*num_samples):]

            ch_val_diag = []
            for j_2 in range(len(ch_val)):
                sig_diag = np.array(np.diag(ch_val[j_2]), copy=True)
                ch_val_diag.append(sig_diag)
            ch_val_diag = np.array(ch_val_diag)

            ch_val_test_diag = []
            for j_2 in range(len(ch_val_test)):
                sig_test_diag = np.array(np.diag(ch_val_test[j_2]), copy=True)
                ch_val_test_diag.append(sig_test_diag)
            ch_val_test_diag = np.array(ch_val_test_diag)


            t_time_dn = 0
            t_time_wm = 0

            with tf.Session() as sess:
                tf.initialize_all_variables().run()
                a_prev = 0
                ref_wm_rate = 0
                ref_wm_ee = 0
                ref_wm_out = 0
                ref_wm_CUI = 0

                ref_eq_rate = 0
                ref_eq_ee = 0
                ref_eq_out = 0
                ref_eq_CUI = 0

                tx_pow_list = []
                for j_2 in range(len(ch_val)):
                    SINR_wm, tx_pow_wm, out_prob_wm, CUI_wm = sinr_wm(p_t, inter_threshold, num_d2d, 10 ** (avg_val + std_val * ch_val[j_2]))

                    tx_pow_list.append(tx_pow_wm)
                    ref_wm_rate = ref_wm_rate + np.mean(np.log(1 + SINR_wm)[:-1])
                    ref_wm_ee = ref_wm_ee + np.mean(np.log(1 + SINR_wm[:-1]) / (tx_pow_wm[:-1] + p_c))
                    ref_wm_out = ref_wm_out + out_prob_wm
                    ref_wm_CUI = ref_wm_CUI + CUI_wm

                    SINR_eq, OUT_prob_eq, CUI_eq = sinr_eq(p_t, inter_threshold, num_d2d,
                                                   10 ** (avg_val + std_val * ch_val[j_2]))

                    ref_eq_rate = ref_eq_rate + np.mean(np.log(1 + SINR_eq)[:-1])
                    ref_eq_out = ref_eq_out + OUT_prob_eq
                    ref_eq_CUI = ref_eq_CUI + CUI_eq



                tx_pow_list = np.array(tx_pow_list)
                ref_wm_rate = ref_wm_rate/len(ch_val)
                ref_wm_ee = ref_wm_ee/len(ch_val)
                ref_wm_out = ref_wm_out / len(ch_val)
                ref_eq_rate = ref_eq_rate / len(ch_val)
                ref_eq_out = ref_eq_out / len(ch_val)




                for i in range(tot_epoch_phase_1):
                    rand_perm = np.random.permutation(len(ch_val))
                    ch_val[:] = ch_val[rand_perm]
                    ch_val_test[:] = ch_val_test[rand_perm]
                    tx_pow_list[:] = tx_pow_list[rand_perm]
                    ch_val_diag[:] = ch_val_diag[rand_perm]
                    ch_val_test_diag[:] = ch_val_test_diag[rand_perm]

                    a = 0

                    for start, end in zip(range(0, len(ch_val), batch_size), range(batch_size, len(ch_val), batch_size)):

                        feed_vec = ch_val[start:end]
                        tx_pow_test = tx_pow_list[start:end, :-1]

                        feed_vec = feed_vec.reshape(-1, (num_d2d+1)**2)

                        sess.run(train_op_3, feed_dict={X: feed_vec, p_keep_conv: 1.0, avg_val_dnn: avg_val, std_val_dnn: std_val, ref_t:tx_pow_test, lr:0.001})
                        a = a + sess.run(cost_3, feed_dict={X: feed_vec, p_keep_conv: 1.0, avg_val_dnn: avg_val, std_val_dnn: std_val, ref_t:tx_pow_test})


                    if i%10 == 0:
                        print("First i = ", i)
                        print("cost = %f, improvement = %f in percentage = %f"%(a/1000, (a_prev-a)/1000, (a_prev-a)/a))
                        a_prev = a
                        pow_temp = sess.run(py_x_t, feed_dict={X: feed_vec, p_keep_conv: 1.0,
                                                       avg_val_dnn: avg_val, std_val_dnn: std_val})

                        print("pow_temp = ", pow_temp[0])
                        print("**" * 20)

                a_prev, b_prev, c_prev, d_prev = 0, 0, 0, 0
                for i in range(tot_epoch):

                    rand_perm = np.random.permutation(len(ch_val))
                    ch_val[:] = ch_val[rand_perm]
                    ch_val_test[:] = ch_val_test[rand_perm]
                    tx_pow_list[:] = tx_pow_list[rand_perm]
                    ch_val_diag[:] = ch_val_diag[rand_perm]
                    ch_val_test_diag[:] = ch_val_test_diag[rand_perm]


                    for start, end in zip(range(0, len(ch_val), batch_size), range(batch_size, len(ch_val), batch_size)):
                        feed_vec = ch_val[start:end]
                        feed_diag = ch_val_diag[start:end]
                        feed_vec = feed_vec.reshape(-1, (num_d2d+1)**2)

                        sess.run(train_op_1,
                                 feed_dict={X: feed_vec, S_Diag: feed_diag, p_keep_conv: 0.9, avg_val_dnn: avg_val,
                                            std_val_dnn: std_val, lambda_dnn: lam_val, lr: 0.0001})

                    if i%100 == 0:
                        feed_vec = ch_val[:batch_size]
                        feed_vec = feed_vec.reshape(-1, (num_d2d+1)**2)
                        feed_diag = ch_val_diag[:batch_size]
                        ## Below, a => cost, b => inter_prob, c => EE
                        a = -sess.run(cost_1,
                                      feed_dict={X: feed_vec, S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                 std_val_dnn: std_val, lambda_dnn: lam_val})
                        b = -sess.run(tf_rate,
                                     feed_dict={X: feed_vec, S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_dnn: lam_val})
                        c = sess.run(CUE_inter_prob,
                                      feed_dict={X: feed_vec, S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                 std_val_dnn: std_val, lambda_dnn: lam_val}) + 1e-15
                        d = -sess.run(tf_ee,
                                     feed_dict={X: feed_vec, S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                std_val_dnn: std_val, lambda_dnn: lam_val})

                        e = sess.run(CUE_inter_plot,
                                      feed_dict={X: feed_vec, S_Diag: feed_diag, p_keep_conv: 1.0, avg_val_dnn: avg_val,
                                                 std_val_dnn: std_val, lambda_dnn: lam_val})



                        feed_vec_1 = ch_val_test[:batch_size]
                        feed_vec_1 = feed_vec_1.reshape(-1, (num_d2d+1)**2)
                        feed_diag_1 = ch_val_test_diag[:batch_size]

                        a_1 = -sess.run(cost_1, feed_dict={X: feed_vec_1, S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                           avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_dnn: lam_val})
                        b_1 = -sess.run(tf_rate, feed_dict={X: feed_vec_1, S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                           avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_dnn: lam_val})
                        c_1 = sess.run(CUE_inter_prob, feed_dict={X: feed_vec_1, S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                                   avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_dnn: lam_val}) + 1e-15
                        d_1 = -sess.run(tf_ee, feed_dict={X: feed_vec_1, S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                          avg_val_dnn: avg_val, std_val_dnn: std_val, lambda_dnn: lam_val})

                        e_1 = sess.run(CUE_inter_plot, feed_dict={X: feed_vec_1, S_Diag: feed_diag_1, p_keep_conv: 1.0,
                                                          avg_val_dnn: avg_val, std_val_dnn: std_val})


                        print("second iter: ", i)

                        print("(COST) cost = %f, improvement = %f in percentage = %f" % (
                        a, (a - a_prev), (a - a_prev) / a * 100))

                        print("(RATE) cost = %f, improvement = %f in percentage = %f, WMMSE = %f, EQ = %f" % (
                        b, (b - b_prev), (b - b_prev) / b * 100, ref_wm_rate, ref_eq_rate))

                        print("(Inter_Prob) cost = %f, improvement = %f in percentage = %f, WMMSE = %f, EQ = %f" % (
                        c, (c - c_prev), (c - c_prev) / c * 100, ref_wm_out, ref_eq_out))

                        print("(EE) cost = %f, improvement = %f in percentage = %f, WMMSE = %f" % (
                        d*1e3, (d - d_prev), (d - d_prev) / d * 100, ref_wm_ee*1e3))

                        print("(Inter) cost = %f,  WMMSE = %f, EQ = %f" % (
                            e*1e10, ref_wm_CUI*1e10, ref_eq_CUI*1e10))
                        print("<Test results> cost = %f, rate = %f, Inter = %f EE = %f" % (a_1, b_1, c_1, d_1 * 1e3))

                        a_prev = a
                        b_prev = b
                        c_prev = c
                        d_prev = d

                        print("**" * 20)



                # evaluation phase
                for start, end in zip(range(0, len(ch_val), batch_size), range(batch_size, len(ch_val_test), batch_size)):
                    feed_vec = ch_val_test[start:end]
                    feed_vec = feed_vec.reshape(-1, (num_d2d + 1) ** 2)

                    pw_temp = sess.run(py_x_t, feed_dict={X: feed_vec, p_keep_conv: 1.0,
                                                        avg_val_dnn: avg_val, std_val_dnn: std_val})

                    for j in range(batch_size):
                        ## Calculating the performance of test set
                        feed_vec_temp = ch_val_test[start:end]
                        SINR_dnn_val, OUT_prob_dnn, Inter_temp = sinr_dnn(pw_temp[j], inter_threshold, num_d2d,
                                                                  10 ** (avg_val + std_val * feed_vec_temp[j]))

                        cap_1_DUE = cap_1_DUE + np.mean(np.log2(1 + SINR_dnn_val)[:-1])
                        cap_1_CUE = cap_1_CUE + np.mean(np.log2(1 + SINR_dnn_val)[-1:])
                        out_1 = out_1 + OUT_prob_dnn
                        Inter_1 = Inter_1 + Inter_temp

                        power_1 = power_1 + pw_temp
                        ee_1 = ee_1 + np.mean(np.divide(np.log2(1 + SINR_dnn_val) , (pw_temp + p_c) ))

                        ## Calculating the performance of all equal power case
                        SINR_eq, OUT_prob_eq, Inter_temp = sinr_eq(p_t, inter_threshold, num_d2d,
                                                              10 ** (avg_val + std_val * feed_vec_temp[j]))


                        cap_eq_DUE = cap_eq_DUE + np.mean(np.log2(1 + SINR_eq)[:-1])
                        cap_eq_CUE = cap_eq_CUE + np.mean(np.log2(1 + SINR_eq)[-1:])
                        ee_eq = ee_eq + np.mean(np.log2(1 + SINR_eq)) / ((p_t + p_c))
                        out_eq = out_eq + OUT_prob_eq
                        Inter_eq = Inter_eq + Inter_temp

                for start, end in zip(range(0, len(ch_val), batch_size),
                                      range(batch_size, len(ch_val), batch_size)):

                    feed_vec = ch_val[start:end]
                    feed_vec = feed_vec.reshape(-1, (num_d2d + 1) ** 2)

                    pw_temp = sess.run(py_x_t, feed_dict={X: feed_vec, p_keep_conv: 1.0,
                                                          avg_val_dnn: avg_val, std_val_dnn: std_val})

                    for j in range(batch_size):
                        ## Calculating the performance of test set
                        feed_vec_temp = ch_val[start:end]
                        SINR_dnn_val, OUT_prob_dnn, Inter_temp = sinr_dnn(pw_temp[j], inter_threshold, num_d2d,
                                                              10 ** (avg_val + std_val * feed_vec_temp[j]))

                        cap_2_DUE = cap_2_DUE + np.mean(np.log2(1 + SINR_dnn_val)[:-1])
                        cap_2_CUE = cap_2_CUE + np.mean(np.log2(1 + SINR_dnn_val)[-1:])
                        out_2 = out_2 + OUT_prob_dnn
                        Inter_2 = Inter_2 + Inter_temp

                        power_2 = power_2 + pw_temp
                        ee_2 = ee_2 + np.mean(np.divide(np.log2(1 + SINR_dnn_val), (pw_temp + p_c)))

                        ## Calculating the performance of WMMSE
                        SINR_wm, tx_pow_wm, OUT_prob_wm, Inter_temp = sinr_wm(p_t, inter_threshold, num_d2d,
                                                                  10 ** (avg_val + std_val * feed_vec_temp[j]))

                        cap_wm_DUE = cap_wm_DUE + np.mean(np.log2(1 + SINR_wm)[:-1])
                        cap_wm_CUE = cap_wm_CUE + np.mean(np.log2(1 + SINR_wm)[-1:])
                        ee_wm = ee_wm + np.mean(np.log2(1 + SINR_wm)) / ((p_t + p_c))
                        out_wm = out_wm + OUT_prob_wm
                        Inter_wm = Inter_wm + Inter_temp



                cap_1_DUE, cap_eq_DUE = cap_1_DUE/len(ch_val_test), cap_eq_DUE/len(ch_val_test)
                cap_1_CUE, cap_eq_CUE = cap_1_CUE / len(ch_val_test), cap_eq_CUE / len(ch_val_test)
                out_1, out_eq = out_1 / len(ch_val_test), out_eq / len(ch_val_test)
                Inter_1, Inter_eq = Inter_1 / len(ch_val_test), Inter_eq / len(ch_val_test)

                cap_2_DUE, cap_wm_DUE = cap_2_DUE / len(ch_val), cap_wm_DUE / len(ch_val)
                cap_2_CUE, cap_wm_CUE = cap_2_CUE / len(ch_val), cap_wm_CUE / len(ch_val)
                out_2, out_wm = out_2 / len(ch_val), out_wm / len(ch_val)
                Inter_2, Inter_wm = Inter_2 / len(ch_val_test), Inter_wm / len(ch_val_test)


                print("Rate (DUE): DNN(training) => %0.3f, DNN(test) => %0.3f, WMMSE => %0.3f, EQ => %0.3f" % (
                    cap_2_DUE, cap_1_DUE, cap_wm_DUE, cap_eq_DUE))
                print("Rate (CUE): DNN(training) => %0.3f, DNN(test) => %0.3f, WMMSE => %0.3f, EQ => %0.3f" % (
                    cap_2_CUE, cap_1_CUE, cap_wm_CUE, cap_eq_CUE))
                print("Out: DNN(training) => %0.3f, DNN(test) => %0.3f, WMMSE => %0.3f, EQ => %0.3f" % (
                    out_2, out_1, out_wm, out_eq))
                print("Inter: DNN(training) => %0.3f, DNN(test) => %0.3f, WMMSE => %0.3f, EQ => %0.3f" % (
                    10*math.log10(Inter_2), 10*math.log10(Inter_1), 10*math.log10(Inter_wm), 10*math.log10(Inter_eq)))


            cap_mat_1_DUE[k, :] = cap_mat_1_DUE[k, :] + cap_1_DUE
            cap_mat_1_CUE[k, :] = cap_mat_1_CUE[k, :] + cap_1_CUE
            out_mat_1[k, :] = out_mat_1[k, :] + out_1
            Inter_mat_1[k, :] = Inter_mat_1[k, :] + Inter_1


            cap_mat_2_DUE[k, :] = cap_mat_2_DUE[k, :] + cap_2_DUE
            cap_mat_2_CUE[k, :] = cap_mat_2_CUE[k, :] + cap_2_CUE
            out_mat_2[k, :] = out_mat_2[k, :] + out_2
            Inter_mat_2[k, :] = Inter_mat_2[k, :] + Inter_2

            cap_mat_eq_DUE[k, :] = cap_mat_eq_DUE[k, :] + cap_eq_DUE
            cap_mat_eq_CUE[k, :] = cap_mat_eq_CUE[k, :] + cap_eq_CUE
            out_mat_eq[k, :] = out_mat_eq[k, :] + out_eq
            Inter_mat_eq[k, :] = Inter_mat_eq[k, :] + Inter_eq

            cap_mat_wm_DUE[k, :] = cap_mat_wm_DUE[k, :] + cap_wm_DUE
            cap_mat_wm_CUE[k, :] = cap_mat_wm_CUE[k, :] + cap_wm_CUE
            out_mat_wm[k, :] = out_mat_wm[k, :] + out_wm
            Inter_mat_wm[k, :] = Inter_mat_wm[k, :] + Inter_wm




            print("**" * 40)
            print("   " * 40)
            print("Rate(DUE): EQ  = ", np.transpose(cap_mat_eq_DUE/(l+1)))
            print("Rate(DUE): WMMSE  = ", np.transpose(cap_mat_wm_DUE/(l+1)))
            print("Rate(DUE): dnn(Training) = ", np.transpose(cap_mat_2_DUE/(l+1)))
            print("Rate(DUE): dnn(Test) = ", np.transpose(cap_mat_1_DUE/(l+1)))
            print("**" * 40)
            print("   " * 40)

            print("Rate(CUE): EQ  = ", np.transpose(cap_mat_eq_CUE/(l+1)))
            print("Rate(CUE): WMMSE  = ", np.transpose(cap_mat_wm_CUE/(l+1)))
            print("Rate(CUE): dnn(Training) = ", np.transpose(cap_mat_2_CUE/(l+1)))
            print("Rate(CUE): dnn(Test) = ", np.transpose(cap_mat_1_CUE/(l+1)))
            print("**" * 40)
            print("   " * 40)

            print("Out: EQ  = ", 100*np.transpose(out_mat_eq/(l+1)))
            print("Out: WMMSE  = ", 100*np.transpose(out_mat_wm/(l+1)))
            print("Out: dnn(Training) = ", 100*np.transpose(out_mat_2/(l+1)))
            print("Out: dnn(Test) = ", 100*np.transpose(out_mat_1/(l+1)))
            print("**" * 40)
            print("   " * 40)

            print("Inter: EQ  = ", np.transpose(10*np.log10(Inter_mat_eq[0]/(l+1))))
            print("Inter: WMMSE  = ", np.transpose(10*np.log10(Inter_mat_wm[0]/(l+1))))
            print("Inter: dnn(Training) = ", np.transpose(10*np.log10(Inter_mat_2[0]/(l+1))))
            print("Inter: dnn(Test) = ", np.transpose(10*np.log10(Inter_mat_1[0]/(l+1))))

    cap_mat_1_DUE, cap_mat_2_DUE, cap_mat_eq_DUE, cap_mat_wm_DUE = cap_mat_1_DUE/iter_num, cap_mat_2_DUE/iter_num, cap_mat_eq_DUE/iter_num, cap_mat_wm_DUE/iter_num
    cap_mat_1_CUE, cap_mat_2_CUE, cap_mat_eq_CUE, cap_mat_wm_CUE = cap_mat_1_CUE / iter_num, cap_mat_2_CUE / iter_num, cap_mat_eq_CUE / iter_num, cap_mat_wm_CUE / iter_num
    out_mat_1, out_mat_2, out_mat_eq, out_mat_wm = out_mat_1/iter_num, out_mat_2/iter_num, out_mat_eq/iter_num, out_mat_wm/iter_num
    Inter_mat_1, Inter_mat_2, Inter_mat_eq, Inter_mat_wm = Inter_mat_1/iter_num, Inter_mat_2/iter_num, Inter_mat_eq/iter_num, Inter_mat_wm/iter_num


    return cap_mat_1_DUE, cap_mat_2_DUE, cap_mat_eq_DUE, cap_mat_wm_DUE, cap_mat_1_CUE, cap_mat_2_CUE, cap_mat_eq_CUE, cap_mat_wm_CUE, out_mat_1, out_mat_2, out_mat_eq, out_mat_wm, Inter_mat_1, Inter_mat_2, Inter_mat_eq, Inter_mat_wm



'''
    Channel related parameters
'''
#########################################################################
###############       DO NOT CHANGE            #########################
#########################################################################
# size of bandwith - 10MHz
bw = 10*10**6
p_t_dB = 43.
p_t = 10**(p_t_dB/10)
p_c_dB = 40.
p_c = 10**(p_c_dB/10)
pl_const = 34.5
pl_alpha = 38.
N0W = bw*10**(-174.0/10)   # Noise: -174 dBm/Hz
avg_val = 1
std_val = 1
# size of area sensors are distributed = 500
d2d_dist = 30


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


# generated channel sample
num_samples = 10000

## set parameter
num_d2d = 20
## set Ithre
inter_threshold = 1e3 * N0W

lam_val = 0.996
tot_epoch_phase_1 = 1000
tot_epoch = 1000
iter_num = 1

### Masimize EE  - important, Batch size should be placed in here
batch_size = 200


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
X = tf.placeholder(tf.float32, [batch_size, (num_d2d+1)**2])

# X2 is the reshaped value of X which fascilate the operation
X2 = tf.reshape(X, [-1, num_d2d+1, num_d2d+1])

# S_Diag contains the signal_channel value which fascilate the operation
S_Diag = tf.placeholder(tf.float32, [batch_size, num_d2d+1])

p_keep_conv = tf.placeholder("float")
avg_val_dnn = tf.placeholder("float")
std_val_dnn = tf.placeholder("float")
lambda_dnn = tf.placeholder("float")

## ref_t is the transmit power determined by the WMMSE. We assume that ref_t is later concanated by p_t for cell user
ref_t = tf.placeholder("float", [batch_size, num_d2d, 1])
lr = tf.placeholder("float")

## Define variables. The bias for conv layer is defined at the model function.
w1 = tf.Variable(tf.random_normal(((num_d2d+1)**2, (num_d2d+1)**2), stddev=0.01))
w2 = tf.Variable(tf.random_normal((3, 3, 1, 16), stddev=0.01))
w3 = tf.Variable(tf.random_normal((3, 3, 16, 16), stddev=0.01))
w4 = tf.Variable(tf.random_normal((3, 3, 16, 16), stddev=0.01))
w5 = tf.Variable(tf.random_normal((3, 3, 16, 16), stddev=0.01))
w6 = tf.Variable(tf.random_normal((3, 3, 16, 16), stddev=0.01))
w7 = tf.Variable(tf.random_normal((3, 3, 16, 16), stddev=0.01))
wo = tf.Variable(tf.random_normal((16 * int(math.ceil(num_d2d+1)) * int(math.ceil(num_d2d+1)), num_d2d),
                                  stddev=0.01))
b1 = tf.Variable(tf.random_normal((1, (num_d2d+1)**2), stddev=0.01))
bo = tf.Variable(tf.random_normal((1, num_d2d), stddev=0.01))


#### In here, py_x determine the transmit power of D2D user
py_x = p_t * model(X, w1, w2, w3, w4, w5, w6, w7, wo, b1, bo, p_keep_conv, num_d2d)


py_x_temp = tf.reshape(py_x, [batch_size, num_d2d])
py_x_compare = tf.reshape(py_x, [batch_size, num_d2d, 1])

## pt_cell models the transmit power of CUE
pt_cell = p_t*tf.ones([py_x.get_shape().as_list()[0], 1], tf.float32)

## py_x_t is the concanated transmit power of DUE and CUE
py_x_t = tf.concat([py_x_temp, pt_cell], 1)


sig_pw = tf.multiply(py_x_t, 10**(std_val_dnn*S_Diag+avg_val_dnn))
int_pw_1 = tf.multiply(tf.reshape(py_x_t, [-1, num_d2d+1, 1]), 10**(std_val_dnn*X2+avg_val_dnn))
int_pw_2 = tf.reshape(tf.reduce_sum(int_pw_1, 1), [-1, num_d2d+1])

SINR = tf.div(sig_pw, int_pw_2 - sig_pw + N0W)
cap_val = tf.log(1 + SINR)
CUE_inter = tf.nn.relu((int_pw_2 - sig_pw)[:,-1:] - tf.constant(inter_threshold))
CUE_inter_plot = tf.reduce_mean((int_pw_2 - sig_pw)[:,-1:])


CUE_inter_prob = tf.reduce_mean(tf.cast(tf.not_equal(CUE_inter, 0), tf.float32))
tf_rate = tf.reduce_mean(-cap_val[:,:-1])
tf_ee = tf.reduce_mean(tf.div(-cap_val[:,:-1], py_x_t[:,:-1]+p_c))

cost_1 = (1.0-lambda_dnn)*tf_rate + (lambda_dnn)*tf.reduce_mean(tf.nn.tanh(CUE_inter/N0W/1e6))
train_op_1 = tf.train.AdamOptimizer(lr).minimize(cost_1)

cost_2 = tf.reduce_mean(tf.div(-cap_val, py_x_t+p_c))
train_op_2 = tf.train.AdamOptimizer(lr).minimize(cost_2)

cost_3 = tf.reduce_mean(tf.square(py_x_compare - ref_t))
train_op_3 = tf.train.AdamOptimizer(lr).minimize(cost_3)


######################################################################################33

cap_mat_1_DUE, cap_mat_2_DUE, cap_mat_eq_DUE, cap_mat_wm_DUE, cap_mat_1_CUE, cap_mat_2_CUE, cap_mat_eq_CUE, cap_mat_wm_CUE, out_mat_1, out_mat_2, out_mat_eq, out_mat_wm, Inter_mat_1, Inter_mat_2, Inter_mat_eq, Inter_mat_wm = per_eval(batch_size, inter_threshold, metric=0, perm_val=1)

print("**"*40)
print("   "*40)
print("Rate(DUE) (test)", np.transpose(cap_mat_1_DUE))
print("Rate(DUE) (train)", np.transpose(cap_mat_2_DUE))
print("Rate(DUE) EQ", np.transpose(cap_mat_eq_DUE))
print("Rate(DUE) WMMSE", np.transpose(cap_mat_wm_DUE))
print("   "*40)
print("   "*40)
print("Rate(CUE) (test)", np.transpose(cap_mat_1_CUE))
print("Rate(CUE) (train)", np.transpose(cap_mat_2_CUE))
print("Rate(CUE) EQ", np.transpose(cap_mat_eq_CUE))
print("Rate(CUE) WMMSE", np.transpose(cap_mat_wm_CUE))
print("   "*40)
print("   "*40)
print("Out (test)", 100*np.transpose(out_mat_1))
print("Out (train)", 100*np.transpose(out_mat_2))
print("Out EQ", 100*np.transpose(out_mat_eq))
print("Out WMMSE", 100*np.transpose(out_mat_wm))
print("   "*40)
print("   "*40)
print("Inter (test)", np.transpose(10*np.log10(Inter_mat_1)))
print("Inter (train)", np.transpose(10*np.log10(Inter_mat_2)))
print("Inter EQ", np.transpose(10*np.log10(Inter_mat_eq)))
print("Inter WMMSE", np.transpose(10*np.log10(Inter_mat_wm)))
print("   "*40)
print("   "*40)
