import os
import numpy as np
from scipy import stats

def mse(real_matrix, impute_matrix):
    MSE = ((real_matrix - impute_matrix) ** 2).sum(axis=1).mean(axis=0)
    return MSE

def correlation(real_matrix, impute_matrix):
    real_matrix = real_matrix.reshape(428 * 16383)
    impute_matrix = impute_matrix.reshape(428 * 16383)
    s1, p = stats.pearsonr(real_matrix, impute_matrix)
    return s1

def zero_rate(matrix):
    zeros = len(matrix[matrix == 0])
    return zeros / (matrix.shape[0] * matrix.shape[1])

def load_matrix(root):
    lines = open(root, 'r').readlines()
    header = lines[0]
    matrix = []
    for line in lines[1:]:
        conts = line.strip().split('\t')
        res = [float(i) for i in conts[1:]]
        matrix.append(res)
    
    return np.array(matrix)

if __name__ == '__main__':
    real_matrix = load_matrix('dataset/PDAC/GSM3036911_PDAC-A-ST1-filtered.txt')
    impute_matrix = load_matrix('results/PADC/impute.csv')
    # impute_matrix had padding on 0
    impute_matrix = impute_matrix[:, :-1]

    MSE = mse(real_matrix, impute_matrix)
    pcor = correlation(real_matrix, impute_matrix)
    real_zr = zero_rate(real_matrix)
    impute_zr = zero_rate(impute_matrix)

    print('MSE: %.4f' % (MSE))
    print('pearson correlation: %.4f' % (pcor))
    print('real zero rate: %.4f' % (real_zr))
    print('impute zero rate: %.4f' % (impute_zr))