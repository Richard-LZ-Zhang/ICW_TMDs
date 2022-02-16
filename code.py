import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# # Generate matrices describing the system
M1, K1, L1 = 3.94, 2100, 20
L1_c = 2*(K1*M1)**0.5
wn = (K1/M1)**0.5
fn = wn / 6.28318
l1 = L1 / L1_c
peak_original = 4.597 # with 1/100 mass damper
print("For original system, undamped natural frequency is {}, and damping ratio is {}".format(fn,l1))

hz = np.linspace(0, 5, 1001)
sec = np.linspace(0, 15, 1001)

def construct(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F, print_flag=False):
    row = tmd+1
    M = np.zeros((row, row))
    M[0,0] = M1
    for i in range(1,row):
        M[i,i] = tmd_m[i-1]
    if print_flag:
        print("M:")
        print(M)
    
    L = np.zeros((row,row))
    L[1:,0] = np.array(tmd_l*(-1))
    L[0,1:] = np.array(tmd_l*(-1))
    for i in range(1,row):
        L[i,i] = tmd_l[i-1]
    L[0,0] = L1 + np.sum(tmd_l)
    if print_flag:
        print("L:")
        print(L)
    
    K = np.zeros((row,row))
    K[1:,0] = np.array(tmd_k*(-1))
    K[0,1:] = np.array(tmd_k*(-1))
    for i in range(1,row):
        K[i,i] = tmd_k[i-1]
    K[0,0] = K1 + np.sum(tmd_k)
    if print_flag:
        print("K:")
        print(K)

    F = np.array([F]+[0]*tmd)

    return M, L, K, F

def construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F, print_flag=False):
    row = tmd+1
    length = tmd_l.shape[0]
    # print("Num of plots for different damping ratio: ", length)

    M = np.zeros((row, row))
    M[0,0] = M1
    for i in range(1,row):
        M[i,i] = tmd_m[i-1]
    if print_flag:
        print("M:")
        print(M)
    
    #L is different
    L=np.zeros((length,row,row))
    for i in range(length):
        L[i,:,:] = np.zeros((row,row))
        L[i,1:,0] = np.array(tmd_l[i,:]*(-1))
        L[i,0,1:] = np.array(tmd_l[i,:]*(-1))
        for j in range(1,row):
            L[i,j,j] = tmd_l[i][j-1]
        L[i,0,0] = L1 + np.sum(tmd_l[i,:])
        if print_flag:
            print("L:")
            print(L)
    
    K = np.zeros((row,row))
    K[1:,0] = np.array(tmd_k*(-1))
    K[0,1:] = np.array(tmd_k*(-1))
    for i in range(1,row):
        K[i,i] = tmd_k[i-1]
    K[0,0] = K1 + np.sum(tmd_k)
    if print_flag:
        print("K:")
        print(K)

    F = np.array([F]+[0]*tmd)

    return M, L, K, F

def plot_mul_l(hz, sec, M, L, K, F, damping_ratio, peak_original = 4.787):
    data_len = hz.shape[0]
    length = L.shape[0]
    # The initial of the first f_response as reference amplitude
    f_response_ref = np.abs(freq_response(hz * 2*np.pi, M, L[0,:,:], K, F))[0][0]
    f_response = np.zeros((data_len, length))
    for i in range(length):
        full_response = np.abs(freq_response(hz * 2*np.pi, M, L[i,:,:], K, F))
        f_response[:,i] = full_response[:,0]/f_response_ref
        # normalize according to steady state response for M1

    # Determine suitable legends
    f_legends = (
        'damping ratio {:.4g}: peak {:.4g} at {:.4g} Hz; reduced by {:.4g} percent'.format(
            damping_ratio[i],
            f_response[m][i],
            hz[m],
            (1-(f_response[m,i]/peak_original))*100
        )
        for i, m in enumerate(np.argmax(f_response, axis=0))
    )
    # t_response = time_response(sec, M, L, K, F)
    # equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    # toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    # lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    # t_legends = (
    #     'm{} settled to 2% beyond {:.4g} sec'.format(
    #         i+1,
    #         sec[lastbig[i]]
    #     )
    #     for i, _ in enumerate(t_response.T)
    # )

    # Create plot

    # fig, ax = plt.subplots(2, 1, figsize=(11.0, 7.7))
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Frequency domain response')
    ax.set_xlabel('Frequency/hertz')
    ax.set_ylabel('Amplitude/metre')
    ax.legend(ax.plot(hz, f_response), f_legends)

    # ax[1].set_title('Time domain response')
    # ax[1].set_xlabel('Time/second')
    # ax[1].set_ylabel('Displacement/metre')
    # ax[1].legend(ax[1].plot(sec, t_response), t_legends)

    fig.tight_layout()
    plt.show()
def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )


def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T


def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def interactive(tmd=50, tmd_total_mass=M1/100, damping_ratio=0.01, F1 = 1, peak_original=peak_original):
    tmd_avg_mass = tmd_total_mass / tmd
    w = []
    w.append(np.sqrt(K1/M1))
    tmd_m = np.array([tmd_avg_mass]*tmd)
    reduction = []
    for num in range(1,tmd+1):
        tmd_m = np.array([tmd_avg_mass]*num)
        tmd_k = tmd_m * np.array(w)**2
        tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
        tmd_l = damping_ratio* tmd_l_c[np.newaxis,:]
        M, L, K, F = construct_mul_l(M1, K1, L1, num, tmd_m, tmd_k, tmd_l, F1,print_flag=False)
        data_len = hz.shape[0]
        # The initial of the first f_response as reference amplitude
        f_response = np.abs(freq_response(hz * 2*np.pi, M, L[0,:,:], K, F)[:,0])
        f_response = f_response / f_response[0]
        index = np.argmax(f_response, axis=0)
        w.append(hz[index]*2*np.pi)
        reduction.append(100*(1-f_response[index]/peak_original))
    f_response = np.abs(freq_response(hz * 2*np.pi, M, L[0,:,:], K, F)[:,0])    
    f_response = f_response / f_response[0]
    return f_response, w[:-1], reduction

def main():

    """Main program"""

    # a case with multiple l for damper
    # 0.1, 13.82 percent
    # tmd = 1
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # tmd_m = np.array([tmd_avg_mass]*tmd)
    # tmd_k =  np.array([tmd_k_ref]* tmd)
    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.01, 0.05, 0.1, 0.2, 1, 10])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)


    # # # a case with multiple l for damper, the best damping ratio is 0.09, 14%
    # tmd = 1
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # tmd_m = np.array([tmd_avg_mass]*tmd)
    # tmd_k =  np.array([tmd_k_ref]* tmd)
    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.05, 0.07, 0.09, 0.12, 0.16])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)


    # # a case with 3 dampers with 2 frequencies
    # tmd = 2
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # w2 = 3.4*2*np.pi #Hz
    # tmd_m = np.array([tmd_avg_mass]*tmd)
    # tmd_k =  np.array([tmd_k_ref, w2**2*tmd_avg_mass])
    # f_list = np.sqrt(tmd_k / tmd_m)/ (2*np.pi)
    # print(f_list)
    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.05, 0.09])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)


    # # a case with 3 dampers with 3 frequencies
    # tmd = 3
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # w2 = 3*2*np.pi #Hz
    # w3 = 3.6*2*np.pi
    # tmd_m = np.array([tmd_avg_mass]*tmd)
    # tmd_k =  np.array([tmd_k_ref, w2**2*tmd_avg_mass, w3**2*tmd_avg_mass])
    # f_list = np.sqrt(tmd_k / tmd_m)/ (2*np.pi)
    # print(f_list)
    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.05, 0.09])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)

    # # 0.04, 17.15%
    # tmd = 10
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # tmd_m = np.array([tmd_avg_mass]*tmd)

    # f_list = np.linspace(3.49, 3.85,tmd)
    # tmd_k =  (f_list * 2*np.pi)**2*tmd_avg_mass

    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.04, 0.08, 0.16])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)


    # # now best 0.02 damping, 20.5 percent
    # tmd = 10
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # tmd_m = np.array([tmd_avg_mass]*tmd)

    # f_list = np.linspace(3.3, 4.0,tmd)
    # tmd_k =  (f_list * 2*np.pi)**2*tmd_avg_mass

    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.01, 0.02, 0.04])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)
    
    # # now best 0.03 damping, 12.13 percent
    # tmd = 10
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # tmd_m = np.array([tmd_avg_mass]*tmd)

    # f_list = np.linspace(2.9, 4.4,tmd)
    # tmd_k =  (f_list * 2*np.pi)**2*tmd_avg_mass

    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.02, 0.03, 0.045])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)


    # # The best value is 0.0032, reaching 19.64 percent
    # tmd = 50
    # tmd_total_mass = M1 / 100
    # tmd_avg_mass = tmd_total_mass / tmd
    # tmd_k_ref = K1/M1*tmd_avg_mass
    # tmd_m = np.array([tmd_avg_mass]*tmd)

    # f_list = np.linspace(3.3, 4.0,tmd)
    # tmd_k =  (f_list * 2*np.pi)**2*tmd_avg_mass

    # tmd_l_c = 2*np.sqrt(np.array(tmd_m) * np.array(tmd_k))
    # tmd_ref_damping_ratio = np.array([0.0015,0.005,0.016, 0.032, 0.064, 0.2])
    # tmd_l = tmd_ref_damping_ratio[:,np.newaxis] @ tmd_l_c[np.newaxis,:]
    # # print(tmd_l.shape)
    # # print(tmd_l)
    # F1 = 1
    # M, L, K, F = construct_mul_l(M1, K1, L1, tmd, tmd_m, tmd_k, tmd_l, F1)
    # # print(L_s)
    # # print(L[0,:,:])
    # plot_mul_l(hz, sec, M, L, K, F, tmd_ref_damping_ratio, peak_original = peak_original)

    tmd_list = [10, 25, 50]
    damp_list_list = [[0.01, 0.02, 0.04, 0.08, 0.16], [0.005, 0.01, 0.02, 0.04], [0.01, 0.015, 0.0225, 0.035]]# [[0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.09]*3]
    tmd_total_mass=M1/100
    fig, ax = plt.subplots(3, 3)
    w_list_list = []
    for i in range(0,len(tmd_list)):
        tmd = tmd_list[i]
        damp_list = damp_list_list[i]
        x = np.arange(0,tmd,1)
        w_list = []
        reduction_list = []
        f_response_list = []
        for damp in damp_list:
            f_response, w, reduction = interactive(tmd=tmd, tmd_total_mass= tmd_total_mass, damping_ratio=damp, F1 = 1, peak_original=peak_original)
            ax[i,0].plot(x, np.array(w)/(2*np.pi), label="Damp: {}".format(damp))
            w_list.append(np.array(w)/(2*np.pi))

            ax[i,1].plot(x, reduction, label="Damp: {}".format(damp))
            reduction_list.append(reduction)

            ax[i,2].plot(hz, f_response, label="Damp: {}".format(damp))
            f_response_list.append(f_response)
        print(np.array(reduction_list).min())
        # Create plot

        ax[i,0].set_title('w for {} dampers'.format(tmd))
        ax[i,0].legend()
        # ax[0].legend("Damp: {}".format(damp) for damp in damp_list)
        # ax[0].set_xlabel('Frequency/hertz')
        # ax[0].set_ylabel('Amplitude/metre')
        # ax[0].plot(x, np.array(w_list).T)

        ax[i,1].set_title('Reduction for {} dampers'.format(tmd))
        ax[i,1].legend()
        # ax[1].set_xlabel('Time/second')
        # ax[1].set_ylabel('Displacement/metre')
        # ax[1].plot(x, np.array(reduction_list).T)

        ax[i,2].set_title('Response for {} dampers'.format(tmd))
        ax[i,2].legend()
        w_list_list.append(w_list)
        print(np.array(reduction_list).max())
    plt.show()
    w_distr = sorted(w_list_list[-1][2])
    plt.plot(np.arange(0,50,1), w_distr)
    plt.show()
    

if __name__ == '__main__':
    main()
