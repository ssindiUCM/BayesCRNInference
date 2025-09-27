import numpy as np
import inspect
from scipy import sparse
import scipy
import math

class MatrixExponentialKrylov:

    # krylov method to compute exp(A).b
    def exp_A_x( A, b, basis_size=None):
        """

        :param A: a sparse matrix with csr format
        :param b: a np.vector with dimension n
        :return: exp(A) * b
        """

        t = 0 # check of the time exp(A*0)*b to exp(A*1)*b
        epsilon = 10 ** (-10) # tolerance of the error
        dt = 1
        A_norm = scipy.sparse.linalg.norm(A, 1)  # norma of the matrix A
        # check if A is a zero matrix or not
        if A_norm < 1e-8:
            return b
        if basis_size == None:
            m = adaptively_set_number_of_basis(A_norm*dt) # size of the expansion
            # m = 10 # size of the expansion
            # print('basis_size:', m)
        else:
            m = basis_size

        while t<1:
            h = np.zeros((m+2,m+2))
            V = np.zeros((len(b), m + 1))
            happy_break_down = False

            # Construct the basis; Arnoldi process
            beta = np.linalg.norm(b)
            if beta < 1e-10:
                return np.zeros(b.size) # if b is a zero vector, we return a zero vector
            # print(b)
            # print(b.shape)
            # print(beta)
            # print(V.shape)
            # print(np.ravel(b) / beta)
            V[:, 0:1] = b / beta
            for j in range(m):
                # print(j)
                w = A.dot(V[:, j])
                h_temp = np.dot(w.T, V[:, 0:j+1]).T
                h[0: j+1, j] = h_temp
                w = w - np.dot( V[:, 0: j+1], h_temp)
                w_norm = np.linalg.norm(w)
                # print(V[:, 0:j+1])
                # print(h_temp)
                # print('w:', w)
                h[j+1, j] = w_norm
                if w_norm < 1e-8: # A_norm * epsilon:
                    happy_break_down = True
                    break
                V[:, j+1]= w / w_norm
            if happy_break_down == False :
                h[m + 1, m ] = 1

            # print(h)
            # print(w)
            # print(j)
            # print(V)
            # print(np.linalg.norm(h))

            # Compute the expoential
            error = 1
            # dt = min( 1-t, dt)
            dt = 1- t
            avnorm = np.linalg.norm( A.dot(V[:, m]) )
            while error > epsilon:
                F = scipy.linalg.expm(h * dt)
                if np.isnan(F).any() or np.isinf(F).any(): # if the matrix is nan, we reduce the time step
                    print('Recoganize NaN or inf in computing matrix exponential while using the krylov method. Some action has taken; Don not panick!')
                    dt = dt/2
                    continue


                # estimate the error. it is calculated according to Sidje 1998
                error1 = np.abs( F[m , 0])
                error2 = np.abs( F[m + 1, 0] * avnorm)
                # print('error1, error2:', error1, error2)
                if error1 > 10 * error2:
                    error = error2
                elif error1>error2:
                    error = error2 / (1 - error2 / error1)
                else:
                    error = error1

                if error < epsilon:
                    b = beta*np.dot(V[:,0:m+1 ], F[0:m+1 ,0:1])
                    # print(dt)
                    t = t + dt
                else:
                    dt = dt/2

            # dt = dt * 2

        return b



    # krylov method to compute exp(At).b
    def exp_AT_x( A, T0, Tf, b, basis_size=None):
        """

        :param T0:  initial time
        :param Tf:  final time
        :param b:
        :param basis_size:
        :return:
        """



        t = T0 # check of the time exp(A*0)*b to exp(A*1)*b
        epsilon = 10 ** (-10) # tolerance of the error
        dt = Tf-T0
        A_norm = scipy.sparse.linalg.norm(A, 1) # norma of the matrix A
        # check if A is a zero matrix or not
        if A_norm*dt < 1e-8:
            return [Tf], [b]
        if basis_size == None:
            m = adaptively_set_number_of_basis(A_norm*dt)  # size of the expansion
            # m = 10 # size of the expansion
            # print('basis_size:', m)
        else:
            m = basis_size
        time_list = [] # the result
        result_list = [] # the return

        while t<Tf:
            h = np.zeros((m+2,m+2))
            V = np.zeros((len(b), m + 1))
            happy_break_down = False

            # Construct the basis; Arnoldi process
            beta = np.linalg.norm(b)
            if beta < 1e-10:
                time_list.append(Tf)
                result_list.append(np.zeros(b.size))
                return time_list, result_list # if b is a zero vector, we return a zero vector
            V[:, 0:1] = b / beta
            for j in range(m):
                # print(j)
                w = A.dot(V[:, j])
                h_temp = np.dot(w.T, V[:, 0:j+1]).T
                h[0: j+1, j] = h_temp
                w = w - np.dot( V[:, 0: j+1], h_temp)
                w_norm = np.linalg.norm(w)
                h[j+1, j] = w_norm
                if w_norm < 1e-8: # A_norm * epsilon:
                    happy_break_down = True
                    break
                V[:, j+1]= w / w_norm
            if happy_break_down == False :
                h[m + 1, m ] = 1

            # Compute the expoential
            error = 1
            # dt = min( Tf-t, dt)
            dt = Tf-t
            avnorm = np.linalg.norm(A.dot(V[:, m]))
            while error > epsilon:
                F = scipy.linalg.expm(h * dt)
                if np.isnan(F).any() or np.isinf(F).any(): # if the matrix is nan, we reduce the time step
                    print('Recoganize NaN or inf in computing matrix exponential while using the krylov method. Some action has taken; Don not panick!')
                    dt = dt/2
                    continue

                # estimate the error. it is calculated according to Sidje 1998
                error1 = np.abs( F[m , 0])
                error2 = np.abs( F[m + 1, 0] * avnorm)
                if error1 >= 10 * error2:
                    error = error2
                elif error1>error2:
                    error = error2 / (1 - error2 / error1)
                else:
                    error = error1

                if error < epsilon:
                    b = beta*np.dot(V[:,0:m+1 ], F[0:m+1 ,0:1])
                    # print(dt)
                    t = t + dt
                    time_list.append(t)
                    result_list.append(b)
                else:
                    dt = dt/2

        return time_list, result_list

def adaptively_set_number_of_basis(A_norm):
    if A_norm > 50:
        return 45

    for m in range(5, 40+1, 5):
        if A_norm ** m / math.factorial(m) < 10**(-8):
            break

    return m
