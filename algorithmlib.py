#%% Library
from turtle import xcor
import numpy as np
import os
import scipy
import scipy.io
import matplotlib.pyplot as plt
import mne
import math
import seaborn as sn
from time import *
import pickle
from test_three_type_of_shrinkage_method import shrinkage_method
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
#%%
def get_shrinkage_dataset(dataset: np.ndarray, ):
    
    chan_num, time_len, trial_num = dataset.shape
    lw_dataset = np.zeros((trial_num, chan_num * chan_num))
    rblw_dataset = np.zeros_like(lw_dataset)
    ss_dataset = np.zeros_like(lw_dataset)
    ora_dataset = np.zeros_like(lw_dataset)
    emprical_dataset = np.zeros_like(lw_dataset)
    
    for i in range(trial_num):
        shrinkage_par = shrinkage_method(dataset[...,i])
        _, lw_data = shrinkage_par.ledoit_wolf()
        _, rblw_data = shrinkage_par.rao_blackwell_LW()
        _, ss_data = shrinkage_par.schafe_strimmer()
        _, ora_data = shrinkage_par.oracle()
        location = np.mean(dataset[...,i], axis = 1, keepdims = True)
        centered = (dataset[...,i] - location)
        emprical_data = centered @ centered.T
        
        for j in range(chan_num):
            lw_dataset[i, j*chan_num:(j+1)*chan_num] = lw_data[:, j]
            rblw_dataset[i, j*chan_num:(j+1)*chan_num] = rblw_data[:, j]
            ss_dataset[i, j*chan_num:(j+1)*chan_num] = ss_data[:, j]
            ora_dataset[i, j*chan_num:(j+1)*chan_num] = ora_data[:, j]
            emprical_dataset[i, j*chan_num:(j+1)*chan_num] = emprical_data[:,j]
       
    return lw_dataset, rblw_dataset, ss_dataset, ora_dataset, emprical_dataset

def cal_score(testset_label, predict_label, criterion, TPlabel = 1, TNlabel = 0):
    
    tp = np.size(np.intersect1d(np.nonzero(predict_label == 1), 
                                np.nonzero(testset_label.squeeze() == 1)))
    tn = np.size(np.intersect1d(np.nonzero(predict_label == 0), 
                                np.nonzero(testset_label.squeeze() == 0)))
    fp = np.sum(predict_label) - tp
    fn = np.sum(-predict_label + 1) - tn
    tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), 
                                    np.nonzero(testset_label.squeeze() == 1)))\
        / np.size(np.nonzero(testset_label.squeeze() == 1))
    tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), 
                                    np.nonzero(testset_label.squeeze() == 0)))\
        / np.size(np.nonzero(testset_label.squeeze() == 0))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    acc = np.sum(predict_label == testset_label) / testset_label.size
    F1 = 2 * (precision * recall)/(precision + recall)
    b_acc = (tpr + tnr)/2
    AUC = cal_auc(criterion, testset_label, TPlabel, TNlabel)
    
    return acc, b_acc, AUC, F1, tpr, tnr

def cal_auc(score, target, label_positive, label_negative):

    def trapezoid(x1, x2, y1, y2):

        a = np.abs(x1 - x2)
        b = np.abs(y1 + y2)
        area = a * b /2
  
        return area

    trial_num = score.size

    if trial_num != target.size:

        print('The length of tow inpute vector should be equal')

    P = 0
    N = 0

    for i in range(trial_num):

        if target.squeeze()[i] == label_positive:
            P += 1

        elif target.squeeze()[i] == label_negative:
            N += 1

        else:
            print('Wrong target value')

    score = np.real(score.squeeze())
    target = np.real(target.squeeze())
    L = np.vstack((score, target)).T
    idx_L = np.argsort(L[:,0])
    L = L[idx_L[idx_L.size::-1],:]

    fp , fp_pre, tp, tp_pre = 0, 0, 0, 0

    score_pre = - 100000

    auc = 0
    count = 0
    for i in range(trial_num):
        
        if L[i, 0] != score_pre:

            count += 1
            if count == 1:
                curve = np.array([fp/N, tp/P, L[i, 0]])

            else:
                curve = np.vstack((curve , np.array([fp/N, tp/P, L[i, 0]])))

            auc = auc + trapezoid(fp, fp_pre, tp, tp_pre)

            score_pre = L[i, 0]

            tp_pre = tp
            fp_pre = fp
            
        
        if L[i, 1] == label_positive:
            tp += 1
        else:
            fp += 1
    
    curve = np.vstack((curve,np.array([1, 1, 0])))
    auc = auc /(P*N)
    auc = auc + trapezoid(1, fp_pre/N, 1, tp_pre/P)
    return auc

def score(predict_label, criterion, testset_label):
    tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), np.nonzero(testset_label.squeeze() == 1)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 1))
    tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), np.nonzero(testset_label.squeeze() == 0)))\
                                    /np.size(np.nonzero(testset_label.squeeze() == 0))
    AUC = cal_auc(criterion, testset_label, 1, 0)
    return (tpr + tnr)/2, AUC, tpr, tnr

def train_DCPM(trainset, trainset_label, component_idx = 0):
    """
    --------------------------------------------------------------------
    FunctionName:	train_SKDCPM
    Purpose:		train discriminative canonical pattern matching model,
                    witch Sw matrix was estamited by SOA numerical shrinkage
                    method
    Parameter:		
                    1 trainset: ndarray [time, channel, trial]
                    2 trainset_label: [1, trial]

    Return:			1 model: dict(): 5*keys       5*values
                            {key}              {value}
                           1)'filter' :               [channel, channel]
                           2)'component denote':      eig value [,channel]
                           3)'component index':       Index,eig victor used
                           4)'template of target':    [time, channel]
                           5)'template of nontarget': [time, chanel]
                           
    Note:      	    1 library demand : numpy
                    2 you need to set 'denote_threshold', which detmermine 
                      how many eig victor will be used '
                    3 you need to make sure that function 'SOA_shrinkage' 
                      is exist in your code
    --------------------------------------------------------------------
    """
    # denote_threshold = 0.80             # define denote of component

    train_trial = np.zeros(trainset.shape)

    # data centralization
    for i in range(trainset.shape[2]):

        train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 0)
    # get class trial
    class_type = np.unique(trainset_label)
    target = trainset[..., trainset_label.squeeze() == class_type[1].astype(int)]
    nontarget = trainset[..., trainset_label.squeeze() == class_type[0].astype(int)]

    # get class template
    template_tar = np.mean(target, axis = 2)       # extract target template
    template_nontar = np.mean(nontarget, axis = 2) # extract nontarget template
    template_all = (template_tar + template_nontar) / 2

    # calcute  between-class divergence matrix
    sigma = ((template_tar - template_all).T @ (template_tar - template_all) \
            + (template_nontar - template_all).T @ (template_nontar - template_all))/2 
    
    # calcute intra-class divergence matrix
    cov_all2 = np.zeros([target.shape[1], target.shape[1], target.shape[2]])
    cov_all3 = np.zeros([nontarget.shape[1], nontarget.shape[1], nontarget.shape[2]])

    for n in range(target.shape[2]):

        cov_all2[..., n] = (target[..., n].squeeze() - template_tar).T @ (target[..., n].squeeze() - template_tar)

    cov_0 = np.mean(cov_all2, axis = 2)

    for n in range(nontarget.shape[2]):

        cov_all3[..., n] = (nontarget[..., n].squeeze() - template_nontar).T @ (nontarget[..., n].squeeze() - template_nontar)

    cov_1 = np.mean(cov_all3, axis = 2)
    
    sigma2 = (cov_0 + cov_1)/2

    # solve the optimizatino problem
    aim = np.linalg.pinv(sigma2) @ sigma
    #left_vector, svd_value, right_vector =  np.linalg.svd(aim)
    svd_value , right_vector = np.linalg.eig(aim)
    denote_idx = np.argsort(svd_value)
    denote_idx = np.flip(denote_idx)
    sorted_V = svd_value[denote_idx]
    sorted_W = right_vector[:,denote_idx]

    # calcute the filter 
    '''
    component_idx = 0
    component_denote = np.zeros(sorted_V.size)

    for i in range(sorted_V.size+1):

        component_denote = np.sum(sorted_V[0:i]) / np.sum(sorted_V)

        if component_denote >= denote_threshold:           
            component_idx = i
            break
    '''
    # save DCPM model
    model = dict()
    model['filter'] = np.real(sorted_W[:, 0:component_idx]) 
    model['component denote'] = sorted_V
    model['component index'] = component_idx 
    model['template of target'] = template_tar
    model['template of nontarget'] = template_nontar
    return model

def test_DCPM(testset, testset_label, model, comp_num):
    """
    ----------------------------------------------------------------
    FunctionName:	test_DCPM
    Purpose:		test DCPM or SKDCPM mode, get the classification 
                    result
    Parameter:		
                    1 testset: ndarray[time, channel, trial]
                    2 label: [1, trial]
    Return:			accuracy: ndarray[,time]
    Note:      	    1 library demand : numpy
    ----------------------------------------------------------------
    """
    # extract model information
    W = model['filter'] 
    component_idx = model['component index']
    template_tar = model['template of target'] 
    template_nontar =  model['template of nontarget']
    V = model['component denote']
    DSP_filter = W[:, 0:comp_num] 
    
    # get class template
    template_tar = template_tar -  np.tile(np.mean(template_tar, axis = 0),\
                                           (template_tar.shape[0], 1))
    template_nontar = template_nontar - np.tile(np.mean(template_nontar, axis = 0),\
                                                (template_nontar.shape[0], 1))

    # classification
    template_1 = template_tar @ DSP_filter
    template_0 = template_nontar @ DSP_filter

    trial = np.zeros((testset.shape[:1]))
    criterion = np.zeros((testset_label.size))

    for i in range(testset.shape[2]):

        # centraliazation
        trail = testset[..., i].squeeze() - np.mean(testset[..., i], axis = 0)
        # spatial filter
        filtered_trial = trail @ DSP_filter

        if DSP_filter.shape[1] == 1:

            criterion[i] = np.mean((np.cov((np.real(template_0) - np.real(filtered_trial)).T))
                            -(np.cov((np.real(template_1) - np.real(filtered_trial)).T)))
        else:
            criterion[i] = np.mean(np.diag(np.cov((np.real(template_0) - np.real(filtered_trial)).T))
                            -np.diag(np.cov((np.real(template_1) - np.real(filtered_trial)).T)))

    # statistic classification accuracy
    predict_label = (np.sign(criterion) + 1) / 2
    tp = np.size(np.intersect1d(np.nonzero(predict_label == 1), np.nonzero(testset_label.squeeze() == 1)))
    tn = np.size(np.intersect1d(np.nonzero(predict_label == 0), np.nonzero(testset_label.squeeze() == 0)))
    fp = np.sum(predict_label) - tp
    fn = np.sum(-predict_label + 1) -tn
    tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), np.nonzero(testset_label.squeeze() == 1)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 1))
    tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), np.nonzero(testset_label.squeeze() == 0)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 0))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    acc = np.sum(predict_label == testset_label) / testset_label.size
    F1 = 2 * (precision * recall)/(precision + recall)
    b_acc = (tpr + tnr)/2

    AUC = cal_auc(criterion, testset_label, 1, 0)
    return acc, b_acc, AUC, F1, tpr, tnr

def train_DCPM_noise_matrix(trainset_o, trainset_label, cmp_num = 2, shrinkage = True):
    
    # denote_threshold = 0.80             # define denote of component
    trainset = trainset_o[100:,...]
    restset = trainset_o[0:100,...]
    target_rest = restset[..., trainset_label == 1]
    nontarget_rest = restset[..., trainset_label == 0]
    train_trial = np.zeros(trainset.shape)

    # data centralization
    for i in range(trainset.shape[2]):

        train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 0)
    # get class trial
    class_type = np.unique(trainset_label)
    target = trainset[..., trainset_label.squeeze() == class_type[1].astype(int)]
    nontarget = trainset[..., trainset_label.squeeze() == class_type[0].astype(int)]

    # get class template
    template_tar = np.mean(target, axis = 2)       # extract target template
    template_nontar = np.mean(nontarget, axis = 2) # extract nontarget template
    template_all = (template_tar + template_nontar) / 2

    # calcute  between-class divergence matrix
    sigma = ((template_tar - template_all).T @ (template_tar - template_all) \
            + (template_nontar - template_all).T @ (template_nontar - template_all))/2 
    
    # calcute intra-class divergence matrix
    cov_all2 = np.zeros([target.shape[1], target.shape[1], target.shape[2]])
    cov_all3 = np.zeros([nontarget.shape[1], nontarget.shape[1], nontarget.shape[2]])

    for n in range(target.shape[2]):
        cov_rest = target_rest[..., n].T @ target_rest[..., n]
        cov_all2[..., n] = target[..., n].squeeze() .T @ target[..., n].squeeze() - template_tar.T @ template_tar + cov_rest

    cov_0 = np.mean(cov_all2, axis = 2)

    for n in range(nontarget.shape[2]):
        cov_rest = nontarget_rest[..., n].T @ nontarget_rest[..., n]
        cov_all3[..., n] = nontarget[..., n].squeeze().T @ nontarget[..., n].squeeze() - template_nontar.T @ template_nontar + cov_rest

    cov_1 = np.mean(cov_all3, axis = 2)
    sigma2 = (cov_0 + cov_1)/2

    if shrinkage == True:
        Sw_pre = sigma2
        P = Sw_pre.shape[1]
        F = np.trace(Sw_pre)/P
        Tar = F * (np.eye(Sw_pre.shape[0]))
        shrink = shrinkage_method(trainset, Sw_pre, Tar)

        alpha, _ = shrink.oracle()
        
    Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]))
    

    # solve the optimizatino problem
    aim = np.linalg.pinv(Sw) @ sigma
    svd_value , right_vector = np.linalg.eig(aim)
    denote_idx = np.argsort(svd_value)
    denote_idx = np.flip(denote_idx)
    sorted_V = svd_value[denote_idx]
    sorted_W = right_vector[:,denote_idx]

    # save DCPM model
    model = dict()
    model['filter'] = np.real(sorted_W[:, 0:cmp_num]) 
    model['component denote'] = sorted_V
    model['component index'] = cmp_num
    model['template of target'] = template_tar
    model['template of nontarget'] = template_nontar
    return model
 
def train_SKGDCPM(trainset, trainset_label, p = 2, shrinkage = True, component_num = 2, maximum_iteration = 60):
    """
    --------------------------------------------------------------------
    FunctionName:	train_SKGDCPM
    Purpose:		train discriminative canonical pattern matching model,
                    witch Sw matrix was estamited by Snumerical shrinkage
                    method, and Minkowski Distance was used as distance metricsOA 
    Parameter:		
                    1 trainset: ndarray [time, channel, trial]
                    2 trainset_label: [1, trial]
                    3 p: p-norm  int[,]
                    4 maximum_iteration: int

    Return:			1 model: dict(): 4*keys       4*values
                            {key}              {value}
                           1)'filter' :               [channel, channel]
                           2)'template of target':    [time, channel]
                           3)'template of nontarget': [time, chanel]
                           4) 'p' :                   [,]
                           
    Note:      	    1 library demand : numpy / sympy
    --------------------------------------------------------------------
    """
    # get trianset information
    time_len, chan_num, trial_num = trainset.shape
    target_trial = trainset[..., trainset_label.squeeze() == 1]
    nontarget_trial = trainset[..., trainset_label.squeeze() == 0]
    target_trial_num = target_trial.shape[2]
    nontarget_trial_num = nontarget_trial.shape[2]
    template_target = np.mean(target_trial, axis = 2)
    template_nontarget = np.mean(nontarget_trial, axis = 2)
    template_all = (template_target + template_nontarget) / 2

    # initialize iteration condition for interation 1
    W = np.zeros((chan_num, component_num))
    B_t = np.diag(np.ones((chan_num, )))
    warning_flag = 0
    warning_ceriterion_register = list()
    warning_component_idx_register = list()

    componet_criterion_register = list()
    # iteration process1
    for c in range(component_num):

        # projection data to the nullspace of existing spatial filter
        B = B_t
        ftrainset = np.einsum('pct, cx -> pxt', trainset, B)
        target_trial = ftrainset[..., trainset_label.squeeze() == 1]
        nontarget_trial = ftrainset[..., trainset_label.squeeze() == 0]
        template_target = np.mean(target_trial, axis = 2)
        template_nontarget = np.mean(nontarget_trial, axis = 2)
        template_all = (template_target + template_nontarget) / 2

        # initialize iteration condition for interation 2
        stop_criterion = 1 * (10**(-2))
        episilon = 1 * (10**(-20))
        criterion_register = list()
        criterion_register.append(stop_criterion + 1)
        maximum_iteration = 300

        w = np.ones((B.shape[1], ))/np.linalg.norm(np.ones((B.shape[1], )), ord = 2)
        wt_register = list()
        wt_register.append(w)

        Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
        Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
        Z = np.dstack((Z_tar, Z_ntar))
        V_tar = template_target - template_all
        V_ntar = template_nontarget - template_all
   
        # calcute the SOA shrinkage parameter
        if (c == 0) & (shrinkage == True):

            sum_all = 0
            for j in range(trial_num):
                sum_all = sum_all + (Z[..., j].T / np.abs(Z[..., j] @ w)**(2-p)) @ Z[..., j]

            Sw = sum_all/(trial_num* time_len)
            n= trial_num * time_len
            P = Sw.shape[1]
            F = np.trace(Sw)/P
            alpha_soa =((-1/P * np.trace(Sw @ Sw) + np.trace(Sw)**2)) / ((n-1)/P * (np.trace(Sw @ Sw) - (np.trace(Sw)**2)/P))
            theta = (np.trace(Sw**2)-np.trace(Sw)**2/p)/(np.trace(Sw**2)+((1-2)/p)*np.trace(Sw)**2)
            alpha_soa1 =1/(((n+1-2)/p)*theta)
            alpha = np.min([alpha_soa, 1])

        elif (c == 0) & (shrinkage == False):
            
            alpha = 0
            F = 0
        

        # iteration process2
        for i in range(maximum_iteration):
            
            if criterion_register[i] >= stop_criterion:

                # calcute H(t)
                sum_all = 0
                for j in range(trial_num):
                    sum_all = sum_all + (Z[..., j].T / np.abs(Z[..., j] @ w)**(2-p)) @ Z[..., j]
                #sum_all1 = 0
                #for j in range(trial_num):
                #    for k in range(time_len):
                #        sum_all1 = sum_all1 + (Z[np.newaxis, k, :,j].T / np.abs(Z[np.newaxis, k, :, j] @ w)**(2-p)) @ Z[np.newaxis, k, :, j]
                Ht = (1 - alpha) * sum_all / (trial_num*time_len) + \
                     alpha * F * np.eye(sum_all.shape[0]) * (np.diag(1 / np.abs(w+episilon)**(2-p)))
                # calcute h(t)
                sum_0 = 0 
                for j in range(time_len):
                    sum_0 = sum_0 + \
                            np.abs(V_ntar[j, ...] @ w) ** (p-1) * V_ntar[j, ...] * np.sign(V_ntar[j, ...] @ w)
                sum_1 = 0 
                for j in range(time_len):
                    sum_1 = sum_1 + \
                            np.abs(V_tar[j, ...] @ w) ** (p-1) * V_tar[j, ...] * np.sign(V_tar[j, ...] @ w)                    
                ht = (sum_0 + sum_1)/(2*time_len)
                #  calcute spatial filter vector
                wt = np.linalg.inv(Ht) @ ht / (ht @ np.linalg.inv(Ht) @ ht)
                wt = wt / np.linalg.norm(wt, ord = 2)
                wt_register.append(wt)
                # refresh stop criterion
                criterion_register.append(np.linalg.norm(wt - w, ord = 2))
                # refresh spatial filter vector
                w = wt
            
            else:

                break
        
        # select the spatial filter corresponding with the minium criterion when beyound the maximum iterion number
        if i == maximum_iteration-1:

            criterion_register.pop(0)
            wt_register.pop(0)
            min_iterition_idx =  np.argmin(np.array(criterion_register))
            w = wt_register[min_iterition_idx]
            warning_flag += 1
            warning_ceriterion_register.append(criterion_register[min_iterition_idx])        
            warning_component_idx_register.append(c + 1)
        componet_criterion_register.append(criterion_register)
        # add new spatial filter vector to the spatial filter matrix
        W[:, c] = (B @ w)/ np.linalg.norm(B @ w, ord = 2)

        # calcute the null space of the existing spatial filter
        B_t = scipy.linalg.null_space(W[:, 0:c+1].T)
        
        #M = Matrix(W[:, 0:c+1])
        #M_null = (M.T).nullspace()
        #print(W[:, 0:c+1].T @ B_t)
        #M_array = np.array(M_null).astype(np.float64).squeeze()
        #B_t1 = scipy.linalg.orth(M_array.T)
        #print(W[:, 0:c+1].T @ B_t1)
        #B = sympy.GramSchmidt(M_null)
        #B_t2 = np.array(B).astype(np.float64).squeeze(). T
        #print(W[:, 0:c+1].T @ B_t2)
        k = 1

    # casting the warning when algorithm dose't convergence
    if warning_flag > 0:

        warning_info = "Warning: SOA-SKGDCPM algorithm dosen't convergence in " \
               + str(warning_flag) + " iteritions at p =" + str(p)
        detail = "\t The stop criterion is "+ str(stop_criterion) + " ,but when iterition stop the cerition is:"
        print(warning_info)
        print(detail)

        for i in range(len(warning_ceriterion_register)):
            print('\t|cmp_idx = ' + str(warning_component_idx_register.pop())+'\t|' + str(warning_ceriterion_register.pop()))

    # save the SKGDCPM model
    template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
    template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
    model = dict()
    model['filter'] = W
    model['template of target'] = template_target
    model['template of nontarget'] = template_nontarget
    model['p'] = p
    model['loss function'] = componet_criterion_register
    model['shrinkage cofficient'] = alpha
    model['component index'] = component_num
    return model

def test_SKGDCPM(testset, testset_label, model):
    """
    ----------------------------------------------------------------
    FunctionName:	test_DCPM
    Purpose:		test DCPM or SKDCPM mode, get the classification 
                    result
    Parameter:		
                    1 testset: ndarray[time, channel, trial]
                    2 label: [1, trial]
    Return:			accuracy: ndarray[,time]
    Note:      	    1 library demand : numpy
    ----------------------------------------------------------------
    """
    def p_norm(matrix, p = 1):
        
        time_len, chan_num = matrix.shape
        p_norm_value = 0
        for i in range(time_len):
            for j in range(chan_num):
                p_norm_value = p_norm_value + np.abs(matrix[i,j])**p
        
        return p_norm_value**(1/p)

    # extract model information
    W = model['filter'] 
    template_tar = model['template of target'] 
    template_nontar =  model['template of nontarget']
    p = model['p']
    DSP_filter = W
    
    # get class template
    template_tar = template_tar -  np.tile(np.mean(template_tar, axis = 0),\
                                           (template_tar.shape[0], 1))
    template_nontar = template_nontar - np.tile(np.mean(template_nontar, axis = 0),\
                                                (template_nontar.shape[0], 1))

    # classification
    template_1 = template_tar @ DSP_filter
    template_0 = template_nontar @ DSP_filter

    trial = np.zeros((testset.shape[:1]))
    criterion = np.zeros((testset_label.size))

    for i in range(testset.shape[2]):

        # centraliazation
        trail = testset[..., i].squeeze() - np.mean(testset[..., i], axis = 0)
        # spatial filter
        filtered_trial = trail @ DSP_filter

        dist_ntar = np.real(template_0) - np.real(filtered_trial)
        dist_tar = np.real(template_1) - np.real(filtered_trial)
        criterion[i] = p_norm(dist_ntar, p) - p_norm(dist_tar, p)

    # statistic classification accuracy
    predict_label = (np.sign(criterion) + 1) / 2
    acc = np.sum(predict_label == testset_label) / testset_label.size
    tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), np.nonzero(testset_label.squeeze() == 1)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 1))
    tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), np.nonzero(testset_label.squeeze() == 0)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 0))
    b_acc = (tpr + tnr)/2
    return b_acc, tpr, tnr

def train_JSSDCPM(trainset, trainset_label,  shrinkage = True, component_num = 1, p = 2):
    """
    --------------------------------------------------------------------
    FunctionName:	train_JSSDCPM
    Purpose:		train joint sparse shrinkage discriminative canonical 
                    pattern matching model, witch Sw matrix was estamited
                    by SOA numerical shrinkage method and L2，1 norm was 
                    used as distane metric

    Parameter:		
                    1 trainset: ndarray [channel, time, trial]
                    2 trainset_label: [1, trial]
                    3 p: p-norm  int[,]
                    4 maximum_iteration: int

    Return:			1 model: dict(): 4*keys       4*values
                            {key}              {value}
                           1)'filter' :               [channel, channel]
                           2)'template of target':    [time, channel]
                           3)'template of nontarget': [time, chanel]
                           4) 'p' :                   [,]
                           
    Note:      	    1 library demand : numpy / sympy
    --------------------------------------------------------------------
    """
    # get trianset information
    chan_num, time_len, trial_num = trainset.shape
    target_trial = trainset[..., trainset_label.squeeze() == 1]
    nontarget_trial = trainset[..., trainset_label.squeeze() == 0]
    target_trial_num = target_trial.shape[2]
    nontarget_trial_num = nontarget_trial.shape[2]
    template_target = np.mean(target_trial, axis = 2)
    template_nontarget = np.mean(nontarget_trial, axis = 2)
    template_all = (template_target + template_nontarget) / 2

    Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
    Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
    Z = np.dstack((Z_tar, Z_ntar))
    V_tar = template_target - template_all
    V_ntar = template_nontarget - template_all
    V = np.dstack((V_tar, V_ntar))
    fZ1 = Z
    
    # initialize iteration condition for interation 1
    diag_W = np.diag(np.ones((chan_num, )))
    W = diag_W
    stop_criterion = 10**(-5) * 5
    maximum_iteration = 40
    xi = 10**(-20)
    iteration_num = 0
    wt_register = list()
    criterion_register = list()

    ## calcute the bridge matrices of L2 norm and L2,1 norm
    A1 = 1 / (np.linalg.norm(Z, ord = 2, axis = 0)** (2-p) + xi )
    xx = np.linalg.norm(W, ord = 2, axis = 1)** (2-p)
    xx[xx == np.inf] = 10**20
    A2 = np.diag(1/(xx + xi) ) 
    A3 = 1 / (np.linalg.norm(V, ord = 2, axis = 0)** (2-p) + xi)

    ## calcute the Sw and Sb matrix
    Swp = np.zeros((chan_num, chan_num, trial_num))
    for i in range(trial_num):
        Swp[:, :, i] = Z[..., i] @ np.diag(A1[..., i]) @ Z[..., i].T
    Sw_pre = np.mean(Swp, axis = 2)

    Sbp = np.zeros((chan_num, chan_num, 2))
    for i in range(2):
        Sbp[:, :, i] = V[:, :, i] @ np.diag(A3[..., i]) @ V[:, :, i].T
    Sb = np.mean(Sbp, axis = 2)
    
    ## calcute the shrinkage coefficient and shrinkage Sw
    if shrinkage == True:
        if p == 1:
            
            n = trial_num * time_len
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha_soa =((-1/P * np.trace(Sw_pre @ Sw_pre) + np.trace(Sw_pre)**2)) / ((n-1)/P * (np.trace(Sw_pre @ Sw_pre) - (np.trace(Sw_pre)**2)/P))
            theta = (np.trace(Sw_pre**2)-np.trace(Sw_pre)**2/p)/(np.trace(Sw_pre**2)+((1-2)/p)*np.trace(Sw_pre)**2)
            alpha_soa1 =1/(((n+1-2)/p)*theta)
            alpha = np.min([alpha_soa1, 1])
            
        else:
            n = trial_num * time_len
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha_soa =((-1/P * np.trace(Sw_pre @ Sw_pre) + np.trace(Sw_pre)**2)) / ((n-1)/P * (np.trace(Sw_pre @ Sw_pre) - (np.trace(Sw_pre)**2)/P))
            alpha_soa2 = P/((n-1))
            alpha = np.min([alpha_soa, 1])
    else:

        P = Sw_pre.shape[1]
        F = np.trace(Sw_pre)/P
        alpha = 0
    #alpha = 0.04
    Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ A2)
        
    # iteration 
    for i in range(maximum_iteration):

        # calcute new spatial filter
        aim = np.linalg.pinv(Sb) @ Sw
        svd_value , right_vector = scipy.linalg.eig(Sw, Sb)
        denote_idx = np.argsort(svd_value)
        #denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W0 = right_vector[:,denote_idx]
        sorted_W = sorted_W0[:, 0:component_num]


        # appling new spatial filter
        ftarget_trial = np.einsum('cpt, ca -> apt', target_trial, sorted_W)
        fnontarget_trial = np.einsum('cpt, ca -> apt', nontarget_trial, sorted_W)

        ftemplate_target = np.mean(ftarget_trial, axis = 2)
        ftemplate_nontarget = np.mean(fnontarget_trial, axis = 2)
        ftemplate_all = (ftemplate_target + ftemplate_nontarget ) / 2
        Z_tar = ftarget_trial - np.dstack([ftemplate_target] * target_trial_num)
        Z_ntar = fnontarget_trial - np.dstack([ftemplate_nontarget] * nontarget_trial_num)
        fZ2 = np.dstack((Z_tar, Z_ntar))
        V_tar = ftemplate_target - ftemplate_all
        V_ntar = ftemplate_nontarget - ftemplate_all
        fV = np.dstack((V_tar, V_ntar))

        A1 = 1 / (np.linalg.norm(fZ2, ord = 2, axis = 0) ** (2-p) + xi)
        xx = np.linalg.norm(sorted_W, ord = 2, axis = 1)** (2-p)
        xx[xx == np.inf] = 10**20
        A2_diag = np.diag(1 / (xx + xi))
        A3 = 1 / (np.linalg.norm(fV, ord = 2, axis = 0)** (2-p) + xi)

        ## calcute the Sw 
        A1_diag = np.zeros((time_len, time_len, trial_num))
        for i in range(trial_num):
            A1_diag[:, :, i] = np.diag(A1[..., i])
        Swp1 = np.einsum('cpt, pat->cat', Z, A1_diag)
        Swp = np.einsum('cpt, apt->cat', Swp1, Z)
        Sw_pre = np.mean(Swp, axis = 2)
        

        Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ A2_diag)

        ## calcute Sb matrix
        A3_diag = np.zeros((time_len, time_len, 2))
        for i in range(2):
            A3_diag[:, :, i] = np.diag(A3[..., i])
        Sbp1 = np.einsum('cpt, pat->cat', V, A3_diag)
        Sbp2 = np.einsum('cpt, apt->cat', Sbp1, V)
        Sb = np.mean(Sbp2, axis = 2)
        

        wt_register.append(sorted_W)
        Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(fZ2, axis = 0), axis = 0))
        Jt = np.mean(np.linalg.norm(np.linalg.norm(fZ1, axis = 0), axis = 0))
        criterion_register.append(np.abs(Jt_1 - Jt))
        #print(np.abs(Jt_1 - Jt))
        #refresh data
        W = sorted_W
        fZ1 = fZ2

        if np.abs(Jt_1 - Jt) < stop_criterion:
            break
        
    # save the SKGDCPM model
    template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
    template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
    model = dict()
    model['filter'] = np.real(W)
    model['template of target'] = template_target
    model['template of nontarget'] = template_nontarget
    model['loss function'] = criterion_register
    model['shrinkage cofficient'] = alpha

    return model    

def test_JSSDCPM(testset, testset_label, model, comp_num):
    """
    ----------------------------------------------------------------
    FunctionName:	test_DCPM
    Purpose:		test DCPM or SKDCPM mode, get the classification 
                    result
    Parameter:		
                    1 testset: ndarray[time, channel, trial]
                    2 label: [1, trial]
    Return:			accuracy: ndarray[,time]
    Note:      	    1 library demand : numpy
    ----------------------------------------------------------------
    """
    # extract model information
    W = model['filter'] 
    template_tar = model['template of target'] 
    template_nontar =  model['template of nontarget']
    DSP_filter = W[:, 0:comp_num] 
    
    # get class template
    template_tar = template_tar -  np.tile(np.mean(template_tar, axis = 0),\
                                           (template_tar.shape[0], 1))
    template_nontar = template_nontar - np.tile(np.mean(template_nontar, axis = 0),\
                                                (template_nontar.shape[0], 1))

    # classification
    template_1 = template_tar.T @ DSP_filter
    template_0 = template_nontar.T @ DSP_filter

    trial = np.zeros((testset.shape[:1]))
    criterion = np.zeros((testset_label.size))

    for i in range(testset.shape[2]):

        # centraliazation
        trail = testset[..., i].squeeze() - np.mean(testset[..., i], axis = 0)
        # spatial filter
        filtered_trial = trail @ DSP_filter

        if DSP_filter.shape[1] == 1:

            criterion[i] = np.mean((np.cov((np.real(template_0) - np.real(filtered_trial)).T))
                            -(np.cov((np.real(template_1) - np.real(filtered_trial)).T)))
        else:
            criterion[i] = np.mean(np.diag(np.cov((np.real(template_0) - np.real(filtered_trial)).T))
                            -np.diag(np.cov((np.real(template_1) - np.real(filtered_trial)).T)))

    # statistic classification accuracy
    predict_label = (np.sign(criterion) + 1) / 2
    tp = np.size(np.intersect1d(np.nonzero(predict_label == 1), np.nonzero(testset_label.squeeze() == 1)))
    tn = np.size(np.intersect1d(np.nonzero(predict_label == 0), np.nonzero(testset_label.squeeze() == 0)))
    fp = np.sum(predict_label) - tp
    fn = np.sum(-predict_label + 1) -tn
    tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), np.nonzero(testset_label.squeeze() == 1)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 1))
    tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), np.nonzero(testset_label.squeeze() == 0)))\
                                /np.size(np.nonzero(testset_label.squeeze() == 0))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    acc = np.sum(predict_label == testset_label) / testset_label.size
    F1 = 2 * (precision * recall)/(precision + recall)
    b_acc = (tpr + tnr)/2

    AUC = cal_auc(criterion, testset_label, 1, 0)
    return acc, b_acc, AUC, F1, tpr, tnr

def forward_sparse_constrain_DSP_multi_dimension(trainset, trainset_label ,component_num=2, p = 1, shrinkage = True, Lambda = 0.01):
 
    # get trianset information
    chan_num, time_len, trial_num = trainset.shape
    target_trial = trainset[..., trainset_label.squeeze() == 1]
    nontarget_trial = trainset[..., trainset_label.squeeze() == 0]
    target_trial_num = target_trial.shape[2]
    nontarget_trial_num = nontarget_trial.shape[2]
    template_target = np.mean(target_trial, axis = 2)
    template_nontarget = np.mean(nontarget_trial, axis = 2)
    template_all = (template_target + template_nontarget) / 2
    
    Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
    Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
    Z = np.dstack((Z_tar, Z_ntar))
    V_tar = template_target - template_all
    V_ntar = template_nontarget - template_all
    V = np.dstack((V_tar, V_ntar))
    fZ1 = Z
    
    # initialize iteration condition for interation 1
    stop_criterion = 10**(-7) * 5
    maximum_iteration = 200
    xi = 10**(-20)
    wt_register = list()
    criterion_register = list()
    diag_W = np.diag(np.ones((chan_num, )))
    W = diag_W

    Cov_set_x = np.einsum('cpt, apt-> cat', trainset, trainset)
    #Cov_x = SOA_shrinkage(Cov_set_x, time_len)
    Cov_x = np.mean(Cov_set_x, axis = 2)/time_len
    
    ftrainset = np.einsum('cpt, ca->apt', trainset, W[:, 0:component_num])
    Cov_set_s = np.einsum('cpt, apt-> cat', ftrainset, ftrainset)
    Cov_s = np.mean(Cov_set_s, axis = 2)/time_len

    A = Cov_x @ W[:, 0:component_num] #@ np.linalg.inv(Cov_s)
    A_row =  np.linalg.norm(A, ord = 2, axis = 1)** (2-p)
    D1 = np.diag(1/(A_row + xi))

    ## calcute the bridge matrices of L2 norm and L2,1 norm
    W_row = np.linalg.norm(W, ord = 2, axis = 1)** (2-p)
    W_row [W_row  == np.inf] = 10**20
    D2 = np.diag(1/(W_row  + xi) ) 
    
    ## calcute the Sw and Sb matrix
    Swp = np.zeros((chan_num, chan_num, trial_num))
    for i in range(trial_num):
        Swp[:, :, i] = Z[..., i] @  Z[..., i].T
    Sw_pre = np.mean(Swp, axis = 2)/time_len

    Sbp = np.zeros((chan_num, chan_num, 2))
    for i in range(2):
        Sbp[:, :, i] = V[:, :, i] @  V[:, :, i].T
    Sb = np.mean(Sbp, axis = 2)/time_len
    
    ## calcute the shrinkage coefficient and shrinkage Sw
    if (shrinkage == True):
        if p == 1:
            
            n = trial_num * time_len
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha_soa =((-1/P * np.trace(Sw_pre @ Sw_pre) + np.trace(Sw_pre)**2)) / ((n-1)/P * (np.trace(Sw_pre @ Sw_pre) - (np.trace(Sw_pre)**2)/P))
            alpha_soa2 = P/((n-1))
            alpha = np.min([alpha_soa, 1])
            
        else:
            n = trial_num * time_len
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha_soa =((-1/P * np.trace(Sw_pre @ Sw_pre) + np.trace(Sw_pre)**2)) / ((n-1)/P * (np.trace(Sw_pre @ Sw_pre) - (np.trace(Sw_pre)**2)/P))
            alpha_soa2 = P/((n-1))
            alpha = np.min([alpha_soa, 1])
    elif shrinkage == False:

        P = Sw_pre.shape[1]
        F = np.trace(Sw_pre)/P
        alpha = 0
    #alpha = 0.04
    Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ D2)

    # iteration 
    for i in range(maximum_iteration):
        
        Sw1 = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ D2)
        # calcute new spatial filter
        left_part = (1-Lambda)*Sw1 + Lambda * Cov_x @ D1 @ Cov_x
        aim = np.linalg.pinv(left_part) @ Sb
        svd_value , right_vector = scipy.linalg.eig(Sb, left_part)
        
        if np.sum([svd_value == np.inf + 0j])>2:
            break
        
        svd_value=np.delete(svd_value,np.argwhere([svd_value == np.inf + 0j]))
        denote_idx = np.argsort(np.real(svd_value),)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W0 = np.zeros_like(right_vector)
        for ss in range(denote_idx.size):
            sorted_W0[:,ss] = right_vector[:,denote_idx[ss]]
            
        sorted_W = sorted_W0[:, 0:component_num]

        # appling new spatial filter
        ftarget_trial = np.einsum('cpt, ca -> apt', target_trial, sorted_W)
        fnontarget_trial = np.einsum('cpt, ca -> apt', nontarget_trial, sorted_W)

        ftemplate_target = np.mean(ftarget_trial, axis = 2)
        ftemplate_nontarget = np.mean(fnontarget_trial, axis = 2)
        ftemplate_all = (ftemplate_target + ftemplate_nontarget ) / 2
        Z_tar = ftarget_trial - np.dstack([ftemplate_target] * target_trial_num)
        Z_ntar = fnontarget_trial - np.dstack([ftemplate_nontarget] * nontarget_trial_num)
        fZ2 = np.dstack((Z_tar, Z_ntar))
        V_tar = ftemplate_target - ftemplate_all
        V_ntar = ftemplate_nontarget - ftemplate_all
        fV = np.dstack((V_tar, V_ntar))

        ftrainset = np.einsum('cpt, ca->apt', trainset, sorted_W)
        Cov_set_s = np.einsum('cpt, apt-> cat', ftrainset, ftrainset)
        Cov_s = np.mean(Cov_set_s, axis = 2)/time_len
        
        A = Cov_x @ sorted_W #@ np.linalg.inv(Cov_s)
        A_row =  np.linalg.norm(A, ord = 2, axis = 1)** (2-p)
        D1 = np.diag(1/(A_row + xi))

        xx = np.linalg.norm(sorted_W, ord = 2, axis = 1)** (2-p)
        xx[xx == np.inf] = 10**20
        D2 = np.diag(1 / (xx + xi))

        wt_register.append(sorted_W)
        Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(fZ2, axis = 0), axis = 0))
        Jt = np.mean(np.linalg.norm(np.linalg.norm(fZ1, axis = 0), axis = 0))
        criterion_register.append(np.abs(Jt_1 - Jt))
        #print(np.abs(Jt_1 - Jt))
        #refresh data
        W = sorted_W
        fZ1 = fZ2

        if np.abs(Jt_1 - Jt) < stop_criterion:
            break
            # add new spatial filter vector to the spatial filter matrix
                

    # save the SKGDCPM model
    template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
    template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
    model = dict()
    model['filter'] = np.real(W)
    model['template of target'] = np.swapaxes(template_target, 0, 1)
    model['template of nontarget'] = np.swapaxes(template_nontarget, 0, 1)
    model['loss function'] = criterion_register
    model['component index'] = component_num
    model['component denote'] = sorted_V 
    return model    

def xDAWN_svd(trial_data, sample_frequence ,stimul_time, Ne, component_num =5):

    time_len, chan_num = trial_data.shape
    
    # construct Toepliz matrix
    first_colum = np.zeros((time_len, 1))
    first_colum[np.around(stimul_time * sample_frequence).astype(int), 0] = 1
    first_row = np.zeros((1, int(Ne * sample_frequence)))
    if np.sum(np.around(stimul_time * sample_frequence).astype(int) == 0) >= 1:
        first_row[0, 0] = 1
    D = scipy.linalg.toeplitz(first_colum, first_row )
    
    # calcute the QR fraction of trial_data and topelitz matrix
    Qx, Rx = scipy.linalg.qr(trial_data,  mode='economic')
    Qd, Rd = scipy.linalg.qr(D,  mode='economic')

   
    # calcute the SVD fraction
    left_vector, svd_value, right_vector = scipy.linalg.svd(Qd.T @ Qx)
    lVector = left_vector[:, 0:component_num]
    rVector = right_vector[0:component_num,:].T
    svd_Value = np.diag(svd_value)[0:component_num, 0:component_num]

    # get the spatial filter
    spatial_filter = np.linalg.inv(Rx) @ rVector
    #fdata = np.swapaxes(trial_data @ spatial_filter, 0, 1)

    return spatial_filter 

def generate_dataset(i, s, t, target, nontarget, data_idx):
    
    # load dataset
    dict_name = str(i) + 's' + str(s) + 's' + str(t)
    train_target_idx = data_idx['train target'][dict_name]
    train_ntarget_idx = data_idx['train ntarget'][dict_name]
    test_target_idx = data_idx['test target'][dict_name]
    test_ntarget_idx = data_idx['test ntarget'][dict_name]
    # get trianset and testset
    pre_trainset = np.dstack((target[..., train_target_idx], 
                                nontarget[..., train_ntarget_idx]))
    pre_testset = np.dstack((target[..., test_target_idx], 
                                nontarget[..., test_ntarget_idx]))
    pre_trainset_label = np.hstack((np.ones(train_target_idx.size), 
                                    np.zeros(train_ntarget_idx.size)))
    pre_testset_label = np.hstack((np.ones(test_target_idx.size), 
                                    np.zeros(test_ntarget_idx.size)))
    rand_trainset_idx = np.random.permutation(pre_trainset_label.size)
    rand_testset_idx = np.random.permutation(pre_testset_label.size)
    trainset = pre_trainset[..., rand_trainset_idx]
    trainset_label = pre_trainset_label[..., rand_trainset_idx]
    testset = pre_testset[..., rand_testset_idx]
    testset_label = pre_testset_label[..., rand_testset_idx]
    
    return trainset, trainset_label, testset, testset_label

def transdataform(dataset):
    
    chan_n, time_l, tiral_n = dataset.shape
    catted_data_at_time_dimension = np.reshape(dataset, (chan_n, time_l * tiral_n) , order = "F")
    
    return catted_data_at_time_dimension.T

class SKGDCPM():
    def __init__(
        self, 
        p = 2, 
        shrinkage_type = None, 
        component_num =2, 
        tol=0.01, 
        max_iter = 300
    ):
        self.p = p
        self.shrinkage = shrinkage_type
        self.cmp_num = component_num
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self, trainset, trainset_label):
        """
        --------------------------------------------------------------------
        FunctionName:	train_SKGDCPM
        Purpose:		train discriminative canonical pattern matching model,
                        witch Sw matrix was estamited by SOA numerical shrinkage
                        method, and Minkowski Distance was used as distance metrics
        Parameter:		
                        1 trainset: ndarray [time, channel, trial]
                        2 trainset_label: [1, trial]
                        3 p: p-norm  int[,]
                        4 maximum_iteration: int

        Return:			1 model: dict(): 4*keys       4*values
                                {key}              {value}
                            1)'filter' :               [channel, channel]
                            2)'template of target':    [time, channel]
                            3)'template of nontarget': [time, chanel]
                            4) 'p' :                   [,]
                            
        Note:      	    1 library demand : numpy / sympy
        --------------------------------------------------------------------
        """
        # centerlization
        location_set = trainset.mean(axis = 0, keepdims = True)
        trainset = trainset.copy() - location_set
        # define moduel parameter
        component_num = self.cmp_num
        shrinkage = self.shrinkage
        p = self.p
        stop_criterion = self.tol
        maximum_iteration = self.max_iter
        
        # get trianset information
        time_len, chan_num, trial_num = trainset.shape
        target_trial = trainset[..., trainset_label.squeeze() == 1]
        nontarget_trial = trainset[..., trainset_label.squeeze() == 0]
        target_trial_num = target_trial.shape[2]
        nontarget_trial_num = nontarget_trial.shape[2]
        template_target = np.mean(target_trial, axis = 2)
        template_nontarget = np.mean(nontarget_trial, axis = 2)
        template_all = (template_target + template_nontarget) / 2

        # initialize iteration condition for interation 1
        W = np.zeros((chan_num, component_num))
        B_t = np.diag(np.ones((chan_num, )))
        warning_flag = 0
        warning_ceriterion_register = list()
        warning_component_idx_register = list()

        componet_criterion_register = list()
        # iteration process1
        for c in range(component_num):

            # projection data to the nullspace of existing spatial filter
            B = B_t
            ftrainset = np.einsum('pct, cx -> pxt', trainset, B)
            target_trial = ftrainset[..., trainset_label.squeeze() == 1]
            nontarget_trial = ftrainset[..., trainset_label.squeeze() == 0]
            template_target = np.mean(target_trial, axis = 2)
            template_nontarget = np.mean(nontarget_trial, axis = 2)
            template_all = (template_target + template_nontarget) / 2

            # initialize iteration condition for interation 2
            episilon = 1 * (10**(-20))
            criterion_register = list()
            criterion_register.append(stop_criterion + 1)
            
            w = np.ones((B.shape[1], ))/np.linalg.norm(np.ones((B.shape[1], )), ord = 2)
            wt_register = list()
            wt_register.append(w)

            Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
            Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
            Z = np.dstack((Z_tar, Z_ntar))
            V_tar = template_target - template_all
            V_ntar = template_nontarget - template_all
    
            # calcute the SOA shrinkage parameter
            if (c == 0) & (shrinkage is None):
                alpha = 0
                F = 0

            elif (c == 0) & (shrinkage is not None):
                sum_all = 0
                for j in range(trial_num):
                    sum_all = sum_all + (Z[..., j].T / np.abs(Z[..., j] @ w)**(2-p)) @ Z[..., j]
                Sw = sum_all/(trial_num* time_len)
                
                # n= trial_num * time_len
                # P = Sw.shape[1]
                # F = np.trace(Sw)/P
                # alpha_soa =((-1/P * np.trace(Sw @ Sw) + np.trace(Sw)**2)) / ((n-1)/P * (np.trace(Sw @ Sw) - (np.trace(Sw)**2)/P))
                # alpha1 = np.min([alpha_soa, 1])
                
                P = Sw .shape[0]
                F = np.trace(Sw )/P
                Tar = F * (np.eye(P))
                shrink = shrinkage_method(np.swapaxes(trainset,0,1), Sw, Tar)
                match shrinkage:
                    case "ora" :
                        alpha, _ = shrink.oracle()
                    case "lw" :
                        alpha, _ = shrink.ledoit_wolf()
                    case "rblw":
                        alpha, _ = shrink.rao_blackwell_LW()
                    case "ss":
                        alpha, _ = shrink.schafe_strimmer()
                

            
            # iteration process2
            for i in range(maximum_iteration):
                
                if criterion_register[i] >= stop_criterion:

                    # calcute H(t)
                    sum_all = 0
                    for j in range(trial_num):
                        sum_all = sum_all + (Z[..., j].T / np.abs(Z[..., j] @ w)**(2-p)) @ Z[..., j]
                    #sum_all1 = 0
                    #for j in range(trial_num):
                    #    for k in range(time_len):
                    #        sum_all1 = sum_all1 + (Z[np.newaxis, k, :,j].T / np.abs(Z[np.newaxis, k, :, j] @ w)**(2-p)) @ Z[np.newaxis, k, :, j]
                    Ht = (1 - alpha) * sum_all / (trial_num*time_len) + \
                        alpha * F * np.eye(sum_all.shape[0]) * (np.diag(1 / np.abs(w+episilon)**(2-p)))
                    # calcute h(t)
                    sum_0 = 0 
                    for j in range(time_len):
                        sum_0 = sum_0 + \
                                np.abs(V_ntar[j, ...] @ w) ** (p-1) * V_ntar[j, ...] * np.sign(V_ntar[j, ...] @ w)
                    sum_1 = 0 
                    for j in range(time_len):
                        sum_1 = sum_1 + \
                                np.abs(V_tar[j, ...] @ w) ** (p-1) * V_tar[j, ...] * np.sign(V_tar[j, ...] @ w)                    
                    ht = (sum_0 + sum_1)/(2*time_len)
                    #  calcute spatial filter vector
                    wt = np.linalg.inv(Ht) @ ht / (ht @ np.linalg.inv(Ht) @ ht)
                    wt = wt / np.linalg.norm(wt, ord = 2)
                    wt_register.append(wt)
                    # refresh stop criterion
                    criterion_register.append(np.linalg.norm(wt - w, ord = 2))
                    # refresh spatial filter vector
                    w = wt
                
                else:

                    break
            
            # select the spatial filter corresponding with the minium criterion when beyound the maximum iterion number
            if i == maximum_iteration-1:

                criterion_register.pop(0)
                wt_register.pop(0)
                min_iterition_idx =  np.argmin(np.array(criterion_register))
                w = wt_register[min_iterition_idx]
                warning_flag += 1
                warning_ceriterion_register.append(criterion_register[min_iterition_idx])        
                warning_component_idx_register.append(c + 1)
            componet_criterion_register.append(criterion_register)
            # add new spatial filter vector to the spatial filter matrix
            W[:, c] = (B @ w)/ np.linalg.norm(B @ w, ord = 2)

            # calcute the null space of the existing spatial filter
            B_t = scipy.linalg.null_space(W[:, 0:c+1].T)
            

        # casting the warning when algorithm dose't convergence
        if warning_flag > 0:

            warning_info = "Warning: SOA-SKGDCPM algorithm dosen't convergence in " \
                + str(warning_flag) + " iteritions at p =" + str(p)
            detail = "\t The stop criterion is "+ str(stop_criterion) + " ,but when iterition stop the cerition is:"
            print(warning_info)
            print(detail)

            for i in range(len(warning_ceriterion_register)):
                print('\t|cmp_idx = ' + str(warning_component_idx_register.pop())+'\t|' + str(warning_ceriterion_register.pop()))

        # save the SKGDCPM model
        template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
        template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
        self.filter = W
        self.target_tmp = template_target
        self.nontarget_tmp = template_nontarget
        self.loss_fun = componet_criterion_register
        self.shrinkage_coff = alpha
        return self
    
    def transform(self, dataset, cmp_num):
        time_len, _, trial_num = dataset.shape
        filtered_dataset = np.zeros((time_len, cmp_num, trial_num))
        W = self.filter[:,0:cmp_num]
        for i in range(trial_num):
            filtered_dataset[..., i] = dataset[..., i] @ W
        return filtered_dataset
    
    def predict(self, testset, cmp_num):
        
        # centralization
        location = testset.mean(axis = 0, keepdims = True)
        testset = testset.copy() - location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.target_tmp
        template_nontar =  self.nontarget_tmp
        p = self.p
        DSP_filter = self.filter[:,0:cmp_num]
        # get filtered data
        ftestset = self.transform(testset, cmp_num)
        # get class template
        template_tar = template_tar.copy() -  template_tar.mean(axis = 0, keepdims =True)
        template_nontar = template_nontar.copy() - template_nontar.mean(axis = 0, keepdims =True)

        # classification
        template_1 = template_tar @ DSP_filter
        template_0 = template_nontar @ DSP_filter
        
        self.criterion = np.zeros((trial_num))
        for i in range(trial_num):
            # spatial filter
            filtered_trial = ftestset[..., i].squeeze()
            dist_ntar = np.real(template_0) - np.real(filtered_trial)
            dist_tar = np.real(template_1) - np.real(filtered_trial)
            self.criterion[i] = self._p_norm(dist_ntar, p) - self._p_norm(dist_tar, p)

        # statistic classification accuracy
        self.predict_label = (np.sign(self.criterion) + 1) / 2

        return self.predict_label, self.criterion
    
    def _predict(self, testset, cmp_num):
        
        # centralization
        # location = testset.mean(axis = 0, keepdims = True)
        # testset = testset.copy() - location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.target_tmp
        template_nontar =  self.nontarget_tmp
        p = self.p
        DSP_filter = self.filter[:,0:cmp_num]
        # get filtered data
        ftestset = self.transform(testset, cmp_num)
        # get class template
        # template_tar = template_tar.copy() -  template_tar.mean(axis = 0, keepdims =True)
        # template_nontar = template_nontar.copy() - template_nontar.mean(axis = 0, keepdims =True)
        template_tar = template_tar -  np.tile(np.mean(template_tar, axis = 0),\
                                            (template_tar.shape[0], 1))
        template_nontar = template_nontar - np.tile(np.mean(template_nontar, axis = 0),\
                                                    (template_nontar.shape[0], 1))
        # classification
        template_1 = template_tar @ DSP_filter
        template_0 = template_nontar @ DSP_filter
        
        self.criterion = np.zeros((trial_num))
        for i in range(trial_num):
            # spatial filter
            filtered_trial = ftestset[..., i].squeeze()
            dist_ntar = np.real(template_0) - np.real(filtered_trial)
            dist_tar = np.real(template_1) - np.real(filtered_trial)
            self.criterion[i] = self._p_norm(dist_ntar, p) - self._p_norm(dist_tar, p)

        # statistic classification accuracy
        self.predict_label = (np.sign(self.criterion) + 1) / 2

        return self.predict_label, self.criterion    
    
    def score(self, testset, testset_label, cmp_num):
        predict_label, criterion = self._predict(testset, cmp_num)
        predict_label = (np.sign(criterion) + 1) / 2
        tp = np.size(np.intersect1d(np.nonzero(predict_label == 1), 
                                    np.nonzero(testset_label.squeeze() == 1)))
        tn = np.size(np.intersect1d(np.nonzero(predict_label == 0), 
                                    np.nonzero(testset_label.squeeze() == 0)))
        fp = np.sum(predict_label) - tp
        fn = np.sum(-predict_label + 1) -tn
        tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), 
                                     np.nonzero(testset_label.squeeze() == 1)))\
                                    /np.size(np.nonzero(testset_label.squeeze() == 1))
        tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), 
                                     np.nonzero(testset_label.squeeze() == 0)))\
                                    /np.size(np.nonzero(testset_label.squeeze() == 0))
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        acc = np.sum(predict_label == testset_label) / testset_label.size
        F1 = 2 * (precision * recall)/(precision + recall)
        b_acc = (tpr + tnr)/2
        AUC = cal_auc(criterion, testset_label, 1, 0)
        
        return acc, b_acc, AUC, F1, tpr, tnr
    
    def _p_norm(self, matrix, p = 1):
        
        time_len, chan_num = matrix.shape
        p_norm_value = 0
        for i in range(time_len):
            for j in range(chan_num):
                p_norm_value = p_norm_value + np.abs(matrix[i,j])**p
        
        return p_norm_value**(1/p)

class JSSDCPM():
    
    def __init__(
        self, 
        p = 2, 
        shrinkage_type = None, 
        cov_norm = "l2",
        component_num =2, 
        tol=10**(-5)* 5, 
        max_iter = 40
    ):
        self.p = p
        self.shrinkage = shrinkage_type
        self.cov_norm = cov_norm
        self.cmp_num = component_num
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, trainset, trainset_label):
        """fit JSSDCPM model
        Purpose:		
                        train joint sparse shrinkage discriminative canonical 
                        pattern matching model, witch Sw matrix was estamited
                        by SOA numerical shrinkage method and L2，1 norm was 
                        used as distane metric

        Parameter:		
                        1 trainset: ndarray [channel, time, trial]
                        2 trainset_label: [1, trial]
                        3 p: p-norm  int[,]
                        4 maximum_iteration: int

        Return:			self
        Note:      	    1 library demand : numpy / sympy
        """
        # centerlization
        # location_set = trainset_o.mean(axis = 1, keepdims = True)
        # trainset = trainset_o.copy() - location_set
        
        # define moduel parameter
        component_num = self.cmp_num
        shrinkage = self.shrinkage
        p = self.p
        stop_criterion = self.tol
        maximum_iteration = self.max_iter
        cov_norm = self.cov_norm 
        # get trianset information
        chan_num, time_len, trial_num = trainset.shape
        target_trial = trainset[..., trainset_label.squeeze() == 1].copy()
        nontarget_trial = trainset[..., trainset_label.squeeze() == 0].copy()
        target_trial_num = target_trial.shape[2]
        nontarget_trial_num = nontarget_trial.shape[2]
        template_target = np.mean(target_trial, axis = 2)
        template_nontarget = np.mean(nontarget_trial, axis = 2)
        template_all = (template_target + template_nontarget) / 2

        Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
        Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
        Z = np.dstack((Z_tar, Z_ntar))
        V_tar = template_target - template_all
        V_ntar = template_nontarget - template_all
        V = np.dstack((V_tar, V_ntar))
        fZ1 = Z
        
        # initialize iteration condition for interation 1
        diag_W = np.diag(np.ones((chan_num, )))
        W = diag_W
        xi = 10**(-20)
        iteration_num = 0
        wt_register = list()
        criterion_register = list()

        ## calcute the bridge matrices of L2 norm and L2,1 norm
        A1 = 1 / (np.linalg.norm(Z, ord = 2, axis = 0)** (2-p) + xi )
        xx = np.linalg.norm(W, ord = 2, axis = 1)** (2-p)
        xx[xx == np.inf] = 10**20
        A2 = np.diag(1/(xx + xi) ) 
        A3 = 1 / (np.linalg.norm(V, ord = 2, axis = 0)** (2-p) + xi)
        
        ## covariance type
        match cov_norm:
            case 'lp':
                # calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i] @ np.diag(A1[..., i]) @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i] @ np.diag(A3[..., i]) @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)
                
            case 'l2':
                ## calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i]  @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i]  @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)
            case _:
                return "Something's wrong with the shrinkage norm"
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        if shrinkage is None:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha = 0
        else:

            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0])) @ A2
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            match shrinkage:
                case "ora" :
                    alpha, _ = shrink.oracle()
                case "lw" :
                    alpha, _ = shrink.ledoit_wolf()
                case "rblw":
                    alpha, _ = shrink.rao_blackwell_LW()
                case "ss":
                    alpha, _ = shrink.schafe_strimmer()
            
        Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ A2)
                    
        # iteration 
        for i in range(maximum_iteration):

            # calcute new spatial filter
            try:
                svd_value , right_vector = scipy.linalg.eigh(Sb, Sw)
            except:
                svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
                
            denote_idx = np.argsort(-svd_value) # 从小到大排序
            # denote_idx = np.flip(denote_idx)
            sorted_V = -svd_value[denote_idx]
            sorted_W0 = right_vector[:,denote_idx]
            sorted_W = sorted_W0[:, 0:component_num]

            # appling new spatial filter
            ftarget_trial = np.einsum('cpt, ca -> apt', target_trial, sorted_W)
            fnontarget_trial = np.einsum('cpt, ca -> apt', nontarget_trial, sorted_W)

            ftemplate_target = np.mean(ftarget_trial, axis = 2)
            ftemplate_nontarget = np.mean(fnontarget_trial, axis = 2)
            ftemplate_all = (ftemplate_target + ftemplate_nontarget ) / 2
            Z_tar = ftarget_trial - np.dstack([ftemplate_target] * target_trial_num)
            Z_ntar = fnontarget_trial - np.dstack([ftemplate_nontarget] * nontarget_trial_num)
            fZ2 = np.dstack((Z_tar, Z_ntar))
            V_tar = ftemplate_target - ftemplate_all
            V_ntar = ftemplate_nontarget - ftemplate_all
            fV = np.dstack((V_tar, V_ntar))

            A1 = 1 / (np.linalg.norm(fZ2, ord = 2, axis = 0) ** (2-p) + xi)
            xx = np.linalg.norm(sorted_W, ord = 2, axis = 1)** (2-p)
            xx[xx == np.inf] = 10**20
            A2_diag = np.diag(1 / (xx + xi))
            A3 = 1 / (np.linalg.norm(fV, ord = 2, axis = 0)** (2-p) + xi)

            ## calcute the Sw 
            A1_diag = np.zeros((time_len, time_len, trial_num))
            for i in range(trial_num):
                A1_diag[:, :, i] = np.diag(A1[..., i])
            Swp1 = np.einsum('cpt, pat->cat', Z, A1_diag)
            Swp = np.einsum('cpt, apt->cat', Swp1, Z)
            Sw_pre = np.mean(Swp, axis = 2)
            # Sw_pre = np.mean(Swp[...,0:target_trial_num], axis = 2) + np.mean(Swp[...,target_trial_num:], axis = 2)
            # Sw_pre = Sw_pre/2

            Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ A2_diag)
            self.F = F
            
            ## calcute Sb matrix
            A3_diag = np.zeros((time_len, time_len, 2))
            for i in range(2):
                A3_diag[:, :, i] = np.diag(A3[..., i])
            Sbp1 = np.einsum('cpt, pat->cat', V, A3_diag)
            Sbp2 = np.einsum('cpt, apt->cat', Sbp1, V)
            Sb = np.mean(Sbp2, axis = 2)
            

            wt_register.append(sorted_W)
            Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(fZ2, axis = 0), axis = 0))
            Jt = np.mean(np.linalg.norm(np.linalg.norm(fZ1, axis = 0), axis = 0))
            criterion_register.append(np.abs(Jt_1 - Jt))
            #print(np.abs(Jt_1 - Jt))
            #refresh data
            W = sorted_W
            fZ1 = fZ2

            if np.abs(Jt_1 - Jt) < stop_criterion:
                break
            
        # save the SKGDCPM model
        template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
        template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
        self.filter = np.real(W)
        self.target_tmp = template_target
        self.nontarget_tmp = template_nontarget
        self.loss_fun = criterion_register
        self.shrinkage_coff = alpha
        return self

    def transform(self, dataset, cmp_num):
        """transform origin data to filtered data

        Args:
            dataset (ndarry): chan_num*time_len*trial_num
            cmp_num (int): the number of component used for filter

        Returns:
            ndarry: filtered datatset
        """
        _, time_len, trial_num = dataset.shape
        filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
        W = self.filter[:,0:cmp_num]
        for i in range(trial_num):
            filtered_dataset[..., i] = W.T @ dataset[..., i] 
        return filtered_dataset

    def fit_transform(self, trainset, trainset_label, cmp_num):
        
        self.fit(trainset, trainset_label)
        ftrainset = self.transform(trainset, cmp_num)
        return ftrainset
    
    def predict(self, testset, cmp_num):
        """predict testset label

        Args:
            testset_o (ndarray[channel, time, trial]): tesetset
            cmp_num (int): the filter dimension for predict

        Returns:
            predict_label: 
            criterion:
        """
    
        # centralization
        location = np.mean(testset, axis = 1, keepdims = True)
        testset = testset - location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.target_tmp
        template_nontar =  self.nontarget_tmp
        DSP_filter = self.filter[:,0:cmp_num]
        # get class template
        template_tar = template_tar - template_tar.mean(axis = 1, keepdims =True)
        template_nontar = template_nontar - template_nontar.mean(axis = 1, keepdims =True)
        # get filtered class template
        template_1 =  DSP_filter.T @ template_tar
        template_0 =  DSP_filter.T @ template_nontar
        # get filtered data
        ftestset = self.transform(testset, cmp_num)     
           
        # classification
        self.criterion = np.zeros((trial_num,))
        for i in range(trial_num):
            filtered_trial = ftestset[..., i]
            dist_ntar = np.linalg.norm(template_0 - filtered_trial)**2
            dist_tar = np.linalg.norm(template_1 - filtered_trial)**2
            self.criterion[i] = dist_ntar - dist_tar
            
            # # same with test_JSSDCPM
            # filtered_trial = ftestset[..., i]
            # if DSP_filter.shape[1] == 1:
            #     self.criterion[i] = np.mean((np.cov((np.real(template_0) - np.real(filtered_trial))))
            #                     -(np.cov((np.real(template_1) - np.real(filtered_trial)))))
            # else:
            #     self.criterion[i] = np.mean(np.diag(np.cov((np.real(template_0) - np.real(filtered_trial))))
            #                     -np.diag(np.cov((np.real(template_1) - np.real(filtered_trial)))))
            
        # statistic classification accuracy
        self.predict_label = (np.sign(self.criterion) + 1) / 2
        return self.predict_label, self.criterion
    
    def _test_JSSDCPM(self, testset, cmp_num):
        """predict testset label

        Args:
            testset_o (ndarray[channel, time, trial]): tesetset
            cmp_num (int): the filter dimension for predict

        Returns:
            predict_label: 
            criterion:
        """
        # centralization
        location = np.mean(testset, axis = 1, keepdims = True)
        testset = testset- location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.target_tmp
        template_nontar =  self.nontarget_tmp
        DSP_filter = self.filter[:,0:cmp_num]
        # get class template
        template_tar = template_tar - template_tar.mean(axis = 0, keepdims =True)
        template_nontar = template_nontar - template_nontar.mean(axis = 0, keepdims =True)
        # get filtered class template
        template_1 =  DSP_filter.T @ template_tar
        template_0 =  DSP_filter.T @ template_nontar
        # get filtered data
        ftestset = self.transform(testset, cmp_num)     
        # prediction
        self.criterion = np.zeros((testset_label.size))
        for i in range(testset.shape[2]):

            filtered_trial =  ftestset[..., i]

            if DSP_filter.shape[1] == 1:
                self.criterion[i] = np.mean((np.cov((np.real(template_0) - np.real(filtered_trial))))
                                -(np.cov((np.real(template_1) - np.real(filtered_trial)))))
            else:
                self.criterion[i] = np.mean(np.diag(np.cov((np.real(template_0) - np.real(filtered_trial))))
                                -np.diag(np.cov((np.real(template_1) - np.real(filtered_trial)))))

        # statistic classification accuracy
        self.predict_label = (np.sign(self.criterion) + 1) / 2
        return self.predict_label, self.criterion

    def cal_score(self, testset_label, predict_label, criterion, TPlabel = 1, TNlabel = 0):
        
        tp = np.size(np.intersect1d(np.nonzero(predict_label == 1), 
                                    np.nonzero(testset_label.squeeze() == 1)))
        tn = np.size(np.intersect1d(np.nonzero(predict_label == 0), 
                                    np.nonzero(testset_label.squeeze() == 0)))
        fp = np.sum(predict_label) - tp
        fn = np.sum(-predict_label + 1) - tn
        tpr = np.size(np.intersect1d(np.nonzero(predict_label == 1), 
                                     np.nonzero(testset_label.squeeze() == 1)))\
            / np.size(np.nonzero(testset_label.squeeze() == 1))
        tnr = np.size(np.intersect1d(np.nonzero(predict_label == 0), 
                                     np.nonzero(testset_label.squeeze() == 0)))\
            / np.size(np.nonzero(testset_label.squeeze() == 0))
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        acc = np.sum(predict_label == testset_label) / testset_label.size
        F1 = 2 * (precision * recall)/(precision + recall)
        b_acc = (tpr + tnr)/2
        AUC = cal_auc(criterion, testset_label, TPlabel, TNlabel)
        
        return acc, b_acc, AUC, F1, tpr, tnr
            
    def score(self, testset, testset_label, cmp_num, TPlabel = 1, TNlabel = 0):
        """evaluate jssdcpm model

        Args:
            testset (ndarry[channel, time, trial]): testset
            testset_label (ndarry): label of test data
            cmp_num (int): the filter dimension for test

        Returns:
            tupel: acc, bacc, auc, f1, tpr, tnr 
        """
    
        predict_label, criterion = self.predict(testset = testset, cmp_num = cmp_num)

        return self.cal_score(testset_label, predict_label, criterion, TPlabel, TNlabel)
    
class JSSDCPM_dymatic_shrinkage(JSSDCPM):
    
    def __init__(        
        self, 
        p = 2, 
        shrinkage_type = None, 
        cov_norm = "l2",
        component_num =2, 
        tol=10**(-5)* 5, 
        max_iter = 40
    ):
        super().__init__(
            p, 
            shrinkage_type, 
            cov_norm, 
            component_num, 
            tol, 
            max_iter
    )

    def fit(self, trainset, trainset_label):
        """fit JSSDCPM model
        Purpose:		
                        train joint sparse shrinkage discriminative canonical 
                        pattern matching model, witch Sw matrix was estamited
                        by SOA numerical shrinkage method and L2，1 norm was 
                        used as distane metric

        Parameter:		
                        1 trainset: ndarray [channel, time, trial]
                        2 trainset_label: [1, trial]
                        3 p: p-norm  int[,]
                        4 maximum_iteration: int

        Return:			self
        Note:      	    1 library demand : numpy / sympy
        """
        # centerlization
        # location_set = trainset_o.mean(axis = 1, keepdims = True)
        # trainset = trainset_o.copy() - location_set
        
        # define moduel parameter
        component_num = self.cmp_num
        shrinkage = self.shrinkage
        p = self.p
        stop_criterion = self.tol
        maximum_iteration = self.max_iter
        cov_norm = self.cov_norm 
        # get trianset information
        chan_num, time_len, trial_num = trainset.shape
        target_trial = trainset[..., trainset_label.squeeze() == 1].copy()
        nontarget_trial = trainset[..., trainset_label.squeeze() == 0].copy()
        target_trial_num = target_trial.shape[2]
        nontarget_trial_num = nontarget_trial.shape[2]
        template_target = np.mean(target_trial, axis = 2)
        template_nontarget = np.mean(nontarget_trial, axis = 2)
        template_all = (template_target + template_nontarget) / 2

        Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
        Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
        Z = np.dstack((Z_tar, Z_ntar))
        V_tar = template_target - template_all
        V_ntar = template_nontarget - template_all
        V = np.dstack((V_tar, V_ntar))
        fZ1 = Z
        
        # initialize iteration condition for interation 1
        diag_W = np.diag(np.ones((chan_num, )))
        W = diag_W
        xi = 10**(-20)
        iteration_num = 0
        wt_register = list()
        criterion_register = list()

        ## calcute the bridge matrices of L2 norm and L2,1 norm
        A1 = 1 / (np.linalg.norm(Z, ord = 2, axis = 0)** (2-p) + xi )
        xx = np.linalg.norm(W, ord = 2, axis = 1)** (2-p)
        xx[xx == np.inf] = 10**20
        A2 = np.diag(1/(xx + xi) ) 
        A3 = 1 / (np.linalg.norm(V, ord = 2, axis = 0)** (2-p) + xi)
        
        ## covariance type
        match cov_norm:
            case 'lp':
                # calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i] @ np.diag(A1[..., i]) @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i] @ np.diag(A3[..., i]) @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)
                
            case 'l2':
                ## calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i]  @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i]  @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)
            case _:
                return "Something's wrong with the shrinkage norm"
            
        ## calcute the shrinkage coefficient and shrinkage Sw
        if shrinkage is None:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha = 0
        else:

            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0])) @ A2
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            match shrinkage:
                case "ora" :
                    alpha, _ = shrink.oracle()
                case "lw" :
                    alpha, _ = shrink.ledoit_wolf()
                case "rblw":
                    alpha, _ = shrink.rao_blackwell_LW()
                case "ss":
                    alpha, _ = shrink.schafe_strimmer()
            
        Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ A2)
        
        # iteration 
        for i in range(maximum_iteration):

            # calcute new spatial filter
            svd_value , right_vector = scipy.linalg.eig(Sw, Sb)
            denote_idx = np.argsort(svd_value)
            #denote_idx = np.flip(denote_idx)
            sorted_V = svd_value[denote_idx]
            sorted_W0 = right_vector[:,denote_idx]
            sorted_W = sorted_W0[:, 0:component_num]

            # appling new spatial filter
            ftarget_trial = np.einsum('cpt, ca -> apt', target_trial, sorted_W)
            fnontarget_trial = np.einsum('cpt, ca -> apt', nontarget_trial, sorted_W)

            ftemplate_target = np.mean(ftarget_trial, axis = 2)
            ftemplate_nontarget = np.mean(fnontarget_trial, axis = 2)
            ftemplate_all = (ftemplate_target + ftemplate_nontarget ) / 2
            Z_tar = ftarget_trial - np.dstack([ftemplate_target] * target_trial_num)
            Z_ntar = fnontarget_trial - np.dstack([ftemplate_nontarget] * nontarget_trial_num)
            fZ2 = np.dstack((Z_tar, Z_ntar))
            V_tar = ftemplate_target - ftemplate_all
            V_ntar = ftemplate_nontarget - ftemplate_all
            fV = np.dstack((V_tar, V_ntar))

            A1 = 1 / (np.linalg.norm(fZ2, ord = 2, axis = 0) ** (2-p) + xi)
            xx = np.linalg.norm(sorted_W, ord = 2, axis = 1)** (2-p)
            xx[xx == np.inf] = 10**20
            A2_diag = np.diag(1 / (xx + xi))
            A3 = 1 / (np.linalg.norm(fV, ord = 2, axis = 0)** (2-p) + xi)

            ## calcute the Sw 
            A1_diag = np.zeros((time_len, time_len, trial_num))
            for i in range(trial_num):
                A1_diag[:, :, i] = np.diag(A1[..., i])
            Swp1 = np.einsum('cpt, pat->cat', Z, A1_diag)
            Swp = np.einsum('cpt, apt->cat', Swp1, Z)
            Sw_pre = np.mean(Swp, axis = 2)
            # Sw_pre = np.mean(Swp[...,0:target_trial_num], axis = 2) + np.mean(Swp[...,target_trial_num:], axis = 2)
            # Sw_pre = Sw_pre/2
            
            ## calcute the shrinkage coefficient and shrinkage Sw
            if shrinkage is None:
                P = Sw_pre.shape[1]
                F = np.trace(Sw_pre)/P
                alpha = 0
            else:

                P = Sw_pre.shape[1]
                F = np.trace(Sw_pre)/P
                Tar = F * (np.eye(Sw_pre.shape[0])) @ A2
                shrink = shrinkage_method(trainset, Sw_pre, Tar)
                match shrinkage:
                    case "ora" :
                        alpha, _ = shrink.oracle()
                    case "lw" :
                        alpha, _ = shrink.ledoit_wolf()
                    case "rblw":
                        alpha, _ = shrink.rao_blackwell_LW()
                    case "ss":
                        alpha, _ = shrink.schafe_strimmer()
                
            Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ A2_diag)
            self.F = F
            
            ## calcute Sb matrix
            A3_diag = np.zeros((time_len, time_len, 2))
            for i in range(2):
                A3_diag[:, :, i] = np.diag(A3[..., i])
            Sbp1 = np.einsum('cpt, pat->cat', V, A3_diag)
            Sbp2 = np.einsum('cpt, apt->cat', Sbp1, V)
            Sb = np.mean(Sbp2, axis = 2)
            

            wt_register.append(sorted_W)
            Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(fZ2, axis = 0), axis = 0))
            Jt = np.mean(np.linalg.norm(np.linalg.norm(fZ1, axis = 0), axis = 0))
            criterion_register.append(np.abs(Jt_1 - Jt))
            #print(np.abs(Jt_1 - Jt))
            #refresh data
            W = sorted_W
            fZ1 = fZ2

            if np.abs(Jt_1 - Jt) < stop_criterion:
                break
            
        # save the SKGDCPM model
        template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
        template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
        self.filter = np.real(W)
        self.target_tmp = template_target
        self.nontarget_tmp = template_nontarget
        self.loss_fun = criterion_register
        self.shrinkage_coff = alpha
        return self

    def transform(self, dataset, cmp_num):
        return super().transform(dataset, cmp_num)

    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)
    
    def predict(self, testset, cmp_num):
        return super().predict(testset, cmp_num)
    
    def score(self, testset, testset_label, cmp_num):
        return super().score(testset, testset_label, cmp_num)

class JSSDCPM_regu(JSSDCPM):
    def __init__(
        self, 
        p = 2, 
        alpha = 0, 
        cov_norm = "lp",
        component_num = 2, 
        tol=10**(-5)* 5, 
        max_iter = 40
    ):
        super().__init__(
            p = p, 
            cov_norm = cov_norm,
            component_num = component_num,
            tol = tol, 
            max_iter = max_iter
        )
        self.alpha = alpha

    def fit(self, trainset, trainset_label):
        """fit JSSDCPM model
        Purpose:		
                        train joint sparse shrinkage discriminative canonical 
                        pattern matching model, witch Sw matrix was estamited
                        by SOA numerical shrinkage method and L2,1 norm was 
                        used as distane metric

        Parameter:		
                        1 trainset: ndarray [channel, time, trial]
                        2 trainset_label: [1, trial]
                        3 p: p-norm  int[,]
                        4 maximum_iteration: int

        Return:			self
                            
        Note:      	    1 library demand : numpy / sympy
        """
        # centerlization
        # location_set = trainset_o.mean(axis = 1, keepdims = True)
        # trainset = trainset_o.copy() - location_set
        
        # define moduel parameter
        component_num = self.cmp_num
        p = self.p
        stop_criterion = self.tol
        maximum_iteration = self.max_iter
        cov_norm = self.cov_norm 
        # get trianset information
        chan_num, time_len, trial_num = trainset.shape
        target_trial = trainset[..., trainset_label.squeeze() == 1].copy()
        nontarget_trial = trainset[..., trainset_label.squeeze() == 0].copy()
        target_trial_num = target_trial.shape[2]
        nontarget_trial_num = nontarget_trial.shape[2]
        template_target = np.mean(target_trial, axis = 2)
        template_nontarget = np.mean(nontarget_trial, axis = 2)
        template_all = (template_target + template_nontarget) / 2

        Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
        Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
        Z = np.dstack((Z_tar, Z_ntar))
        V_tar = template_target - template_all
        V_ntar = template_nontarget - template_all
        V = np.dstack((V_tar, V_ntar))
        fZ1 = Z
        
        # initialize iteration condition for interation 1
        diag_W = np.diag(np.ones((chan_num, )))
        W = diag_W
        xi = 10**(-20)
        iteration_num = 0
        wt_register = list()
        criterion_register = list()

        ## calcute the bridge matrices of L2 norm and L2,1 norm
        A1 = 1 / (np.linalg.norm(Z, ord = 2, axis = 0)** (2-p) + xi )
        xx = np.linalg.norm(W, ord = 2, axis = 1)** (2-p)
        xx[xx == np.inf] = 10**20
        A2 = np.diag(1/(xx + xi) ) 
        A3 = 1 / (np.linalg.norm(V, ord = 2, axis = 0)** (2-p) + xi)
        
        ## covariance type
        match cov_norm:
            case 'lp':
                # calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i] @ np.diag(A1[..., i]) @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i] @ np.diag(A3[..., i]) @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)
                
            case 'l2':
                ## calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i]  @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i]  @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)

            case _:
                return "Something's wrong with the shrinkage norm"
            
        Sw =  Sw_pre + self.alpha * np.eye(Sw_pre.shape[0]) @ A2 
        # iteration 
        for i in range(maximum_iteration):

            # calcute new spatial filter
            svd_value , right_vector = scipy.linalg.eig(Sw, Sb)
            denote_idx = np.argsort(svd_value)
            #denote_idx = np.flip(denote_idx)
            sorted_V = svd_value[denote_idx]
            sorted_W0 = right_vector[:,denote_idx]
            sorted_W = sorted_W0[:, 0:component_num]

            # appling new spatial filter
            ftarget_trial = np.einsum('cpt, ca -> apt', target_trial, sorted_W)
            fnontarget_trial = np.einsum('cpt, ca -> apt', nontarget_trial, sorted_W)

            ftemplate_target = np.mean(ftarget_trial, axis = 2)
            ftemplate_nontarget = np.mean(fnontarget_trial, axis = 2)
            ftemplate_all = (ftemplate_target + ftemplate_nontarget ) / 2
            Z_tar = ftarget_trial - np.dstack([ftemplate_target] * target_trial_num)
            Z_ntar = fnontarget_trial - np.dstack([ftemplate_nontarget] * nontarget_trial_num)
            fZ2 = np.dstack((Z_tar, Z_ntar))
            V_tar = ftemplate_target - ftemplate_all
            V_ntar = ftemplate_nontarget - ftemplate_all
            fV = np.dstack((V_tar, V_ntar))

            A1 = 1 / (np.linalg.norm(fZ2, ord = 2, axis = 0) ** (2-p) + xi)
            xx = np.linalg.norm(sorted_W, ord = 2, axis = 1)** (2-p)
            xx[xx == np.inf] = 10**20
            A2_diag = np.diag(1 / (xx + xi))
            A3 = 1 / (np.linalg.norm(fV, ord = 2, axis = 0)** (2-p) + xi)

            ## calcute the Sw 
            A1_diag = np.zeros((time_len, time_len, trial_num))
            for i in range(trial_num):
                A1_diag[:, :, i] = np.diag(A1[..., i])
            Swp1 = np.einsum('cpt, pat->cat', Z, A1_diag)
            Swp = np.einsum('cpt, apt->cat', Swp1, Z)
            Sw_pre = np.mean(Swp, axis = 2)
            # Sw_pre = np.mean(Swp[...,0:target_trial_num], axis = 2) + np.mean(Swp[...,target_trial_num:], axis = 2)
            # Sw_pre = Sw_pre/2

            Sw = Sw_pre + self.alpha * np.eye(Sw_pre.shape[0]) @ A2_diag

            ## calcute Sb matrix
            A3_diag = np.zeros((time_len, time_len, 2))
            for i in range(2):
                A3_diag[:, :, i] = np.diag(A3[..., i])
            Sbp1 = np.einsum('cpt, pat->cat', V, A3_diag)
            Sbp2 = np.einsum('cpt, apt->cat', Sbp1, V)
            Sb = np.mean(Sbp2, axis = 2)
            

            wt_register.append(sorted_W)
            Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(fZ2, axis = 0), axis = 0))
            Jt = np.mean(np.linalg.norm(np.linalg.norm(fZ1, axis = 0), axis = 0))
            criterion_register.append(np.abs(Jt_1 - Jt))
            #print(np.abs(Jt_1 - Jt))
            #refresh data
            W = sorted_W
            fZ1 = fZ2

            if np.abs(Jt_1 - Jt) < stop_criterion:
                break
            
        # save the SKGDCPM model
        template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
        template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
        self.filter = np.real(W)
        self.target_tmp = template_target
        self.nontarget_tmp = template_nontarget
        self.loss_fun = criterion_register

        return self

    def transform(self, dataset, cmp_num):
        return super().transform(dataset, cmp_num)

    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)
    
    def predict(self, testset, cmp_num):
        return super().predict(testset, cmp_num)
    
    def score(self, testset, testset_label, cmp_num):
        return super().score(testset, testset_label, cmp_num)

class fcsDSP(JSSDCPM):
    
    def __init__(
        self, 
        p = 2, 
        shrinkage_type = None, 
        shrinkage_type2 = None,
        cov_norm = "l2",
        component_num =2, 
        Lambda = 0.001,
        tol=10**(-5)* 5, 
        max_iter = 40
    ):
        super().__init__(
            p = p,
            cov_norm=cov_norm,
            component_num=component_num,
        )
        self.shrinkage = shrinkage_type
        self.shrinkage2 = shrinkage_type2
        self.tol = tol
        self.max_iter = max_iter
        self.Lambda = Lambda
        self.filter = None
        self.target_tmp = None
        self.nontarget_tmp = None
        self.loss_fun = None
        self.shrinkage_coff = None
        self.criterion = None
        self.predict_label = None
     
    def fit(self, trainset, trainset_label):
        # centerlization
        # location_set = trainset_o.mean(axis = 1, keepdims = True)
        # trainset = trainset_o.copy() - location_set
        
        # define moduel parameter
        component_num = self.cmp_num
        shrinkage = self.shrinkage
        shrinkage2 = self.shrinkage2
        p = self.p
        stop_criterion = self.tol
        maximum_iteration = self.max_iter
        cov_norm = self.cov_norm 
        Lambda = self.Lambda
        
        
        # get trianset information
        chan_num, time_len, trial_num = trainset.shape
        target_trial = trainset[..., trainset_label.squeeze() == 1]
        nontarget_trial = trainset[..., trainset_label.squeeze() == 0]
        target_trial_num = target_trial.shape[2]
        nontarget_trial_num = nontarget_trial.shape[2]
        template_target = np.mean(target_trial, axis = 2)
        template_nontarget = np.mean(nontarget_trial, axis = 2)
        template_all = (template_target + template_nontarget) / 2
        
        Z_tar = target_trial - np.dstack([template_target] * target_trial_num)
        Z_ntar = nontarget_trial - np.dstack([template_nontarget] * nontarget_trial_num)
        Z = np.dstack((Z_tar, Z_ntar))
        V_tar = template_target - template_all
        V_ntar = template_nontarget - template_all
        V = np.dstack((V_tar, V_ntar))
        fZ1 = Z
        
        # initialize iteration condition for interation 1
        xi = 10**(-20)
        wt_register = list()
        criterion_register = list()
        diag_W = np.diag(np.ones((chan_num, )))
        W = diag_W

        Cov_set_x = np.einsum('cpt, apt-> cat', trainset, trainset)
        #Cov_x = SOA_shrinkage(Cov_set_x, time_len)
        Cov_x = np.mean(Cov_set_x, axis = 2)/time_len
        
        ftrainset = np.einsum('cpt, ca->apt', trainset, W[:, 0:component_num])
        Cov_set_s = np.einsum('cpt, apt-> cat', ftrainset, ftrainset)
        Cov_s = np.mean(Cov_set_s, axis = 2)/time_len
        
        A = Cov_x @ W[:, 0:component_num] 
        A_row =  np.linalg.norm(A, ord = 2, axis = 1)** (2-p)
        D1 = np.diag(1/(A_row + xi))

        ## calcute the bridge matrices of L2 norm and L2,1 norm
        W_row = np.linalg.norm(W, ord = 2, axis = 1)** (2-p)
        W_row [W_row  == np.inf] = 10**20
        D2 = np.diag(1/(W_row  + xi) ) 
        A1 = 1 / (np.linalg.norm(Z, ord = 2, axis = 0)** (2-p) + xi )
        A3 = 1 / (np.linalg.norm(V, ord = 2, axis = 0)** (2-p) + xi)
        
        ## covariance type
        match cov_norm:
            case 'lp':
                # calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i] @ np.diag(A1[..., i]) @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)/time_len    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i] @ np.diag(A3[..., i]) @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)/time_len
                
            case 'l2':
                ## calcute the Sw and Sb matrix
                Swp = np.zeros((chan_num, chan_num, trial_num))
                for i in range(trial_num):
                    Swp[:, :, i] = Z[..., i]  @ Z[..., i].T
                Sw_pre = np.mean(Swp, axis = 2)/time_len    
                # Sw_pre = np.mean(Swp[..., 0:target_trial_num], axis = 2) +\
                #          np.mean(Swp[..., target_trial_num:], axis = 2)
                # Sw_pre = Sw_pre/2

                Sbp = np.zeros((chan_num, chan_num, 2))
                for i in range(2):
                    Sbp[:, :, i] = V[:, :, i]  @ V[:, :, i].T
                Sb = np.mean(Sbp, axis = 2)/time_len
            case _:
                return "Something's wrong with the shrinkage norm"
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        if shrinkage2 is None:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            alpha = 0
        else:

            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0])) @ D2
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            
            match shrinkage2:
                case "ora" :
                    alpha, _ = shrink.oracle()
                case "lw" :
                    alpha, _ = shrink.ledoit_wolf()
                case "rblw":
                    alpha, _ = shrink.rao_blackwell_LW()
                case "ss":
                    alpha, _ = shrink.schafe_strimmer()
            
        Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ D2)
        
        # iteration 
        for i in range(maximum_iteration):
            

            
            Sw1 = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]) @ D2)
            
            if i == 0:
                if shrinkage is None:
                    P = Sw1.shape[1]
                    F = np.trace(Sw1)/P
                    Lambda = 0
                else:

                    P = Sw1.shape[1]
                    F = np.trace(Sw1)/P
                    Tar = Cov_x @ D1 @ Cov_x
                    shrink = shrinkage_method(trainset, Sw_pre, Tar)
                    match shrinkage:
                        case "ora" :
                            Lambda, _ = shrink.oracle()
                        case "lw" :
                            Lambda, _ = shrink.ledoit_wolf()
                        case "rblw":
                            Lambda, _ = shrink.rao_blackwell_LW()
                        case "ss":
                            Lambda, _ = shrink.schafe_strimmer()
            
            # calcute new spatial filter
            left_part = (1-Lambda)*Sw1 + Lambda * Cov_x @ D1 @ Cov_x
            # left_part = (1-2*alpha)*Sw_pre + alpha * Cov_x @ D1 @ Cov_x + alpha*F * (np.eye(Sw_pre.shape[0]) @ D2)
            
            svd_value , right_vector = scipy.linalg.eig(Sb, left_part)
            
            if np.sum([svd_value == np.inf + 0j])>2:
                break
            
            svd_value=np.delete(svd_value,np.argwhere([svd_value == np.inf + 0j]))
            denote_idx = np.argsort(np.real(svd_value),)
            denote_idx = np.flip(denote_idx)
            sorted_V = svd_value[denote_idx]
            sorted_W0 = np.zeros_like(right_vector)
            for ss in range(denote_idx.size):
                sorted_W0[:,ss] = right_vector[:,denote_idx[ss]]
                
            sorted_W = sorted_W0[:, 0:component_num]

            # appling new spatial filter
            ftarget_trial = np.einsum('cpt, ca -> apt', target_trial, sorted_W)
            fnontarget_trial = np.einsum('cpt, ca -> apt', nontarget_trial, sorted_W)

            ftemplate_target = np.mean(ftarget_trial, axis = 2)
            ftemplate_nontarget = np.mean(fnontarget_trial, axis = 2)
            ftemplate_all = (ftemplate_target + ftemplate_nontarget ) / 2
            Z_tar = ftarget_trial - np.dstack([ftemplate_target] * target_trial_num)
            Z_ntar = fnontarget_trial - np.dstack([ftemplate_nontarget] * nontarget_trial_num)
            fZ2 = np.dstack((Z_tar, Z_ntar))
            V_tar = ftemplate_target - ftemplate_all
            V_ntar = ftemplate_nontarget - ftemplate_all
            fV = np.dstack((V_tar, V_ntar))

            ftrainset = np.einsum('cpt, ca->apt', trainset, sorted_W)
            Cov_set_s = np.einsum('cpt, apt-> cat', ftrainset, ftrainset)
            Cov_s = np.mean(Cov_set_s, axis = 2)/time_len
            
            A = Cov_x @ sorted_W #@ np.linalg.inv(Cov_s)
            A_row =  np.linalg.norm(A, ord = 2, axis = 1)** (2-p)
            D1 = np.diag(1/(A_row + xi))

            xx = np.linalg.norm(sorted_W, ord = 2, axis = 1)** (2-p)
            xx[xx == np.inf] = 10**20
            D2 = np.diag(1 / (xx + xi))

            wt_register.append(sorted_W)
            Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(fZ2, axis = 0), axis = 0))
            Jt = np.mean(np.linalg.norm(np.linalg.norm(fZ1, axis = 0), axis = 0))
            criterion_register.append(np.abs(Jt_1 - Jt))
            #print(np.abs(Jt_1 - Jt))
            #refresh data
            W = sorted_W
            fZ1 = fZ2

            if np.abs(Jt_1 - Jt) < stop_criterion:
                break
                # add new spatial filter vector to the spatial filter matrix
                    

        # save the SKGDCPM model
        template_target = np.mean(trainset[..., trainset_label.squeeze() == 1], axis = 2)
        template_nontarget = np.mean(trainset[..., trainset_label.squeeze() == 0], axis = 2)
        self.filter = np.real(W)
        self.target_tmp = template_target
        self.nontarget_tmp = template_nontarget
        self.loss_fun = criterion_register
        self.shrinkage_coff = alpha
        self.regularization_coff = Lambda
        return self
    
    def transform(self, dataset, cmp_num):
        return super().transform(dataset, cmp_num)

    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)
    
    def predict(self, testset, cmp_num):
        return super().predict(testset, cmp_num)
         
    def score(self, testset, testset_label, cmp_num):
        return super().score(testset, testset_label, cmp_num)

class DCPM(JSSDCPM):
    
    def __init__(
        self,
        component_num: int = 2
    ):
        super().__init__(component_num = component_num)
    
    def fit(self, trainset, trainset_label):
        
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        # data centralization
        for i in range(trainset.shape[2]):

            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
            
        # get class trial
        target = trainset[..., trainset_label.squeeze() == 1]
        nontarget = trainset[..., trainset_label.squeeze() == 0]
        # get class template
        template_tar = np.mean(target, axis = 2)       # extract target template
        template_nontar = np.mean(nontarget, axis = 2) # extract nontarget template
        template_all = (template_tar + template_nontar) / 2
        # calcute  between-class divergence matrix
        sigma = ((template_tar - template_all) @ (template_tar - template_all).T \
                + (template_nontar - template_all) @ (template_nontar - template_all).T)/2 
        # calcute intra-class divergence matrix
        cov_all2 = np.zeros([chan_num, chan_num, trial_num])
        cov_all3 = np.zeros([chan_num, chan_num, trial_num])
        for n in range(target.shape[2]):
            cov_all2[..., n] = (target[..., n].squeeze() - template_tar) \
                                @ (target[..., n].squeeze() - template_tar).T
        cov_0 = np.mean(cov_all2, axis = 2)
        for n in range(nontarget.shape[2]):
            cov_all3[..., n] = (nontarget[..., n].squeeze() - template_nontar) \
                                @ (nontarget[..., n].squeeze() - template_nontar).T
        cov_1 = np.mean(cov_all3, axis = 2)
        sigma2 = (cov_0 + cov_1)/2
        # solve the optimizatino problem
        aim = np.linalg.pinv(sigma2) @ sigma
        svd_value , right_vector = np.linalg.eig(aim)
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W = right_vector[:,denote_idx]
        # save DCPM model
        self.filter = np.real(sorted_W)
        self.target_tmp = template_tar
        self.nontarget_tmp = template_nontar
        return self
    
    def transform(self, dataset, cmp_num):
        return super().transform(dataset, cmp_num)

    
    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)
    
    def predict(self, testset, cmp_num):
        return super().predict(testset, cmp_num)
    
    def score(self, testset, testset_label, cmp_num):
        return super().score(testset, testset_label, cmp_num)
  
class xDAWN(JSSDCPM):
    
    def __int__(
        self,
        component_num : int = 2
    ):
        super().__init__(component_num = component_num)
        
    def _transdataform(
        self,
        dataset, 
        sample_frequence, 
        stimul_start_time_in_a_trial
        ):
        
        chan_n, time_l, tiral_n = dataset.shape
        catted_data_at_time_dimension = np.reshape(dataset, (chan_n, time_l * tiral_n) , order = "F")
        data_len = time_l / sample_frequence
        start_time_sequence = np.arange(0, tiral_n) * data_len
        stimul_start_time_sequence = start_time_sequence + stimul_start_time_in_a_trial
        
        return catted_data_at_time_dimension.T, stimul_start_time_sequence

    def fit(
        self, 
        dataset, 
        sample_frequence = None ,
        stimul_start_time_in_a_trial = None, 
        Ne_len = None
        ):

        if (sample_frequence is None) and (stimul_start_time_in_a_trial is None) and (Ne_len is None):
            
            sample_frequence = 1
            stimul_start_time_in_a_trial = 0
            Ne_len = dataset.shape[1]
            
        component_num = self.cmp_num
        trial_data, stimul_time = self._transdataform(dataset, sample_frequence, stimul_start_time_in_a_trial)
        time_len, chan_num = trial_data.shape
        # construct Toepliz matrix
        first_colum = np.zeros((time_len, 1))
        first_colum[np.around(stimul_time * sample_frequence).astype(int), 0] = 1
        first_row = np.zeros((1, int(Ne_len * sample_frequence)))
        if np.sum(np.around(stimul_time * sample_frequence).astype(int) == 0) >= 1:
            first_row[0, 0] = 1
        D = scipy.linalg.toeplitz(first_colum, first_row )
        
        # calcute the QR fraction of trial_data and topelitz matrix
        Qx, Rx = scipy.linalg.qr(trial_data,  mode='economic')
        Qd, Rd = scipy.linalg.qr(D,  mode='economic')

    
        # calcute the SVD fraction
        left_vector, svd_value, right_vector = scipy.linalg.svd(Qd.T @ Qx)
        lVector = left_vector[:, 0:component_num]
        rVector = right_vector[0:component_num,:].T
        svd_Value = np.diag(svd_value)[0:component_num, 0:component_num]

        # get the spatial filter
        spatial_filter = np.linalg.inv(Rx) @ rVector
        self.filter = spatial_filter
        return spatial_filter 
    
    def transform(self, dataset, cmp_num):
        return super().transform(dataset, cmp_num)
    
    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)
    
    def get_filter(self):
        return self.fitler
    
class Cca(JSSDCPM):
    
    def __int__(self, component_num = 2):
        
        self.cmp_num = component_num

    
    def _transdataform(self, trainset_o, trainset_label):
        
        target_t = trainset_o[..., trainset_label == 1]
        nontarget_t = trainset_o[..., trainset_label == 0]
        c_target = transdataform(target_t)
        c_nontarget = transdataform(nontarget_t)
        X = np.concatenate((c_target, c_nontarget), axis = 0)
        
        tmp_target = target_t.mean(axis = 2)
        cat_tmp_target = np.tile(tmp_target, (1, target_t.shape[2]))
        tmp_nontarget = nontarget_t.mean(axis = 2)
        cat_tmp_nontarget = np.tile(tmp_nontarget, (1, nontarget_t.shape[2]))
        Y = np.concatenate((cat_tmp_target.T, cat_tmp_nontarget.T), axis = 0)
        return X, Y
    
    def fit_paper1(self, trainset_o, trainset_label):

        X, Y = self._transdataform(trainset_o, trainset_label)
        return self.fit(X, Y)

    def fit(self, data1, data2):
        try:
            self.cca = CCA(n_components = self.cmp_num, scale=True, max_iter=500, tol=1e-06, copy=True)
            self.cca.fit(data1, data2)
        except:
            try:
                self.cca = CCA(n_components = self.cmp_num, scale=True, max_iter=700, tol=1e-04, copy=True)
                print('tol = 10**-4')
                self.cca.fit(data1, data2)
            except:
                try:
                    self.cca = CCA(n_components = self.cmp_num, scale=True, max_iter=700, tol=1e-02, copy=True)
                    print('tol = 10**-2')
                    self.cca.fit(data1, data2)
                except:
                    self.cca = CCA(n_components = self.cmp_num, scale=True, max_iter=900, tol=5*1e-01, copy=True)
                    print('tol = 1e-01')
                    self.cca.fit(data1, data2)                    
        self.cca.fit(data1, data2)
        self.filter1 = self.cca.x_rotations_
        self.filter2 = self.cca.y_rotations_
        return self
        
    def transform(self, dataset, type = 'class_1' ,cmp_num = 2):
        
        if type == 'class_1':
            _, time_len, trial_num = dataset.shape
            filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
            W = self.filter1[:,0:cmp_num]
            for i in range(trial_num):
                filtered_dataset[..., i] = W.T @ dataset[..., i] 
            return filtered_dataset
        
        if type == 'class_2':
            _, time_len, trial_num = dataset.shape
            filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
            W = self.filter2[:,0:cmp_num]
            for i in range(trial_num):
                filtered_dataset[..., i] = W.T @ dataset[..., i] 
            return filtered_dataset
            
    def get_filter(self):
        return self.fitler1, self.filter2

class STFeature_SKLDA(JSSDCPM):
    
    def __init__(
        self, solver= 'eigen', 
        shrinkage='auto', 
        priors=None, 
        n_components=None, 
        store_covariance=False, 
        tol=0.001, 
        covariance_estimator=None
    ):
        self.lda = LinearDiscriminantAnalysis(
            solver = solver, 
            shrinkage=shrinkage, 
            priors=priors, 
            n_components=n_components, 
            store_covariance=store_covariance, 
            tol=tol, 
            covariance_estimator=covariance_estimator
        )
    
    def _transform_for_lda(self, dataset):
        
        segment_num = 10
        chan_num, time_len, trial_num = dataset.shape
        segment_idx = np.linspace(0, time_len+1, num = segment_num)
        segment_idx_int  = [int(i) for i in segment_idx]
        dataset2 = np.zeros((chan_num, segment_num, trial_num))
        
        for i in range(len(segment_idx_int)-1):
            dataset2[:, i, :] = dataset[:, segment_idx_int[i]:segment_idx_int[i+1], :].mean(axis = 1)
            
        trans_data = np.transpose(dataset2, [2, 1, 0])
        catted_dataset = np.reshape(trans_data,(trial_num, segment_num * chan_num))
        return catted_dataset
        
    def fit(self, X, y):
        Xc = self._transform_for_lda(X)
        self.lda.fit(Xc, y)
        return self
    
    def predict(self, X):
        Xc = self._transform_for_lda(X)
        return self.lda.predict(Xc)
    
    def decision_function(self, X):
        Xc = self._transform_for_lda(X)
        return self.lda.decision_function(Xc)
        
    def score(self, X, y):
        testset_label = y
        predict_label = self.predict(X)
        criterion = self.decision_function(X)
        return super().cal_score(testset_label, predict_label, criterion)


#%%
if __name__ == '__main__':

    metric1 = list()
    metric2 = list()
    metric3 = list()
    metric4 = list()
    metric5 = list()
    metric6 = list()
    metric7 = list()
    
    # load pre rand index
    # pkl_file = open('balanced_data_idx.pkl', 'rb')
    pkl_file = open('parpar1_kaggle_change_trainset_size.pkl', 'rb')
    
    data_idx = pickle.load(pkl_file)
    sub_list = [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25]
    # set the parameter
    trainset_num = [40, 60, 80, 100, 120, 140, 160 , 180, 200, 220]
    p_value = [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    component_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    testset_num = 100
    repeat_num = 10
    num_of_subject = 26

    # set the file 
    subdata_path =  r'F:\\3_项目\\论文一\\数据集\\Kaggle_origin\\'
    npzfile_name = 'sub{sub_idx}.npz'
    # subject_dataset_save = r'F:\6_汇报\汇报17-JSGDCPM\python代码\subject_result_Addtime_len'
    # datainfo
    sfr = 128
    win_len = 0.8
    pre_len = 0.2
    start = np.around(pre_len * sfr).astype('int')
    end = np.around((pre_len + win_len) * sfr).astype('int')
    # initial the matrix to save the result
    parameter_all_jssdcpm = np.zeros((num_of_subject, len(trainset_num), repeat_num, len(p_value), len(component_number), 6))
    parameter_all_dcpm = np.zeros((num_of_subject, len(trainset_num), repeat_num, len(component_number), 6))
    dirlist_sub = os.listdir(subdata_path)
    parameter1 = np.zeros(( repeat_num, 6))
    parameter2 = np.zeros(( repeat_num, 6))
    parameter = np.zeros(( repeat_num, 6))
    # Monte Carlo cross-validationparameter1
    # subject cycle 
    
    for i in range(1):
        i = 8
        print('Now is subject: '+ str(i+1))
        
        mat_path = subdata_path + npzfile_name.format(sub_idx = str(i+1))
        eeg = np.load(mat_path)
        nontarget_trial = np.swapaxes(eeg['nontarget_trial'][...,start:end], 0, 2)
        target_trial = np.swapaxes(eeg['target_trial'][...,start:end], 0, 2)

        # size of trainset cycle 
        for s in range(1):
            s = 2
            #[2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70]
            print('\t trianset number: '+ str(trainset_num[s]))
            # repeat cycle
            for t in range(10):
                
                trainset, trainset_label, testset, testset_label = generate_dataset(i, s, t, target_trial, nontarget_trial, data_idx)
                trainset_o = np.swapaxes(trainset[:,...], 0, 1)
                testset_o = np.swapaxes(testset[:,...], 0, 1)
                
                idx = 1
                model1 = train_DCPM(trainset[:,...], trainset_label, 20)
                metric1.append(test_DCPM(testset[:,...], testset_label, model1, comp_num = idx))
                model2 = DCPM(component_num = 20)
                model2.fit(trainset_o, trainset_label)
                metric2.append(model2.score(testset_o, testset_label, cmp_num = idx)) 
                # print(metric1[1:3])
                # print(metric2[1:3])
                
                # DSP
                ftrainset = model2.transform(trainset_o, cmp_num = idx)
                ftestset = model2.transform(testset_o, cmp_num = idx)
                lda =  STFeature_SKLDA()
                lda.fit(ftrainset[:,::1,:], trainset_label)
                metric3.append(lda.score(ftestset[:,::1,:], testset_label))
                # print(metric3[1:3])
                
                # xDAWN
                xdawn = xDAWN(component_num = 20)
                xdawn.fit(trainset_o[..., trainset_label == 1], 128, 0.15, 0.6)
                ftrainset = xdawn.transform(trainset_o, cmp_num = idx)
                ftestset = xdawn.transform(testset_o, cmp_num = idx)
                # lda =  STFeature_SKLDA()
                lda.fit(ftrainset[:,::1,:], trainset_label)
                metric4.append( lda.score(ftestset[:,::1,:], testset_label))
                # print(metric4[1:3])                
                
                #
                # CCA
                cca = Cca(component_num = 20)
                cca.fit_paper1(trainset_o, trainset_label)
                ftrainset = cca.transform(trainset_o, cmp_num = idx)
                ftestset = cca.transform(testset_o, cmp_num = idx)
                # lda =  STFeature_SKLDA()
                lda.fit(ftrainset[:,::,:], trainset_label)
                metric5.append(lda.score(ftestset[:,::,:], testset_label))
                # print(metric5[1:3])   

                # jssdcpm
                jssdcpm = JSSDCPM(p = 2.5, shrinkage_type='ora', cov_norm='lp', component_num=idx)
                jssdcpm = jssdcpm.fit(trainset_o, trainset_label)  
                metric6.append(jssdcpm.score(testset_o, testset_label, cmp_num = idx))
                
                # JSSDSP
                ftrainset = jssdcpm.transform(trainset_o, cmp_num = idx)
                ftestset = jssdcpm.transform(testset_o, cmp_num = idx)
                # lda =  STFeature_SKLDA()
                lda.fit(ftrainset[:,::1,:], trainset_label)
                metric7.append(lda.score(ftestset[:,::1,:], testset_label))

    ave_metric1, ave_metric2, ave_metric3, ave_metric4, ave_metric5 , ave_metric6, ave_metric7 = 0,0,0,0,0,0, 0
    Metric1 = metric1.copy()
    Metric2 = metric2.copy()
    Metric3 = metric3.copy()
    Metric4 = metric4.copy()
    Metric5 = metric5.copy()
    Metric6 = metric6.copy()
    Metric7 = metric7.copy()
    
    idx = 1
    repeat_num = 10
    for i in range(repeat_num ):
        ave_metric1 += Metric1.pop()[idx]
        ave_metric2 += Metric2.pop()[idx]
        ave_metric3 += Metric3.pop()[idx]
        ave_metric4 += Metric4.pop()[idx]
        ave_metric5 += Metric5.pop()[idx]
        ave_metric6 += Metric6.pop()[idx]
        ave_metric7 += Metric7.pop()[idx]
        
    print('DCPM:',ave_metric1/repeat_num )
    print('DCPM:',ave_metric2/repeat_num )
    print('JSSDCPM',ave_metric6/repeat_num )
    print('DSP:',ave_metric3/repeat_num )
    print('JSSDSP',ave_metric7/repeat_num )
    print('XDAWN',ave_metric4/repeat_num )
    print('CCA',ave_metric5/repeat_num )
    
    
    
    k = 1
                
#%%
if __name__ == '__main__':

    metric1 = list()
    metric2 = list()
    metric3 = list()
    metric4 = list()
    metric5 = list()
    # load pre rand index
    pkl_file = open('balanced_data_idx.pkl', 'rb')
    # pkl_file = open('parpar1_kaggel.pkl', 'rb')
    
    data_idx = pickle.load(pkl_file)
    sub_list = [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25]
    # set the parameter
    trainset_num = [40, 60, 80, 100, 120, 140, 160 , 180, 200, 220]
    p_value = [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    component_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    testset_num = 100
    repeat_num = 10
    num_of_subject = 26

    # set the file 
    subdata_path =  r'F:\6_汇报\汇报16-SKGDCPM\target_nontarget_and_background_data'
    # subject_dataset_save = r'F:\6_汇报\汇报17-JSGDCPM\python代码\subject_result_Addtime_len'

    # initial the matrix to save the result
    parameter_all_jssdcpm = np.zeros((num_of_subject, len(trainset_num), repeat_num, len(p_value), len(component_number), 6))
    parameter_all_dcpm = np.zeros((num_of_subject, len(trainset_num), repeat_num, len(component_number), 6))
    dirlist_sub = os.listdir(subdata_path)
    parameter1 = np.zeros(( repeat_num, 6))
    parameter2 = np.zeros(( repeat_num, 6))
    parameter = np.zeros(( repeat_num, 6))
    # Monte Carlo cross-validationparameter1
    # subject cycle 
    
    for i in range(1):
        i = 11
        print('Now is subject: '+ str(i))
        JSSDCPM_spatial_filterset = dict()
        # mat_path = subdata_path + "\\" + dirlist_sub[i]
        mat_path = subdata_path + "\\" + "{}.mat".format(str(i+1))
        
        eeg = scipy.io.loadmat(mat_path)
        nontarget = eeg['target_trial']
        target = eeg['nontarget_trial']

        # size of trainset cycle 
        for s in range(1):
            s = 9
            #[2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70]
            print('\t trianset number: '+ str(trainset_num[s-1]))
            # repeat cycle
            for t in range(1):
                
                trainset, trainset_label, testset, testset_label = generate_dataset(i, s, t, target, nontarget, data_idx)
                trainset_o = np.swapaxes(trainset[100:260,...], 0, 1)
                testset_o = np.swapaxes(testset[100:260,...], 0, 1)
                
                idx = 2
                model1 = train_DCPM_noise_matrix(trainset[0:260,...], trainset_label,cmp_num = 20)
                metric1.append(test_DCPM(testset[100:260,...], testset_label, model1, comp_num = idx))
                
                print(metric1[-1])
                model2 = train_DCPM(trainset[100:260,...], trainset_label, 20)
                metric2.append( test_DCPM(testset[100:260,...], testset_label, model2, comp_num = idx))
                print(metric2[-1])
                
                jssdcpm = JSSDCPM(p = 2, shrinkage_type='ora', cov_norm='lp', component_num=idx)
                jssdcpm = jssdcpm.fit(trainset_o, trainset_label)  
                metric3.append(jssdcpm.score(testset_o, testset_label, cmp_num = idx))
                print(metric3[-1])
                
                fcsdsp = fcsDSP(p = 1, shrinkage_type='oas', shrinkage_type2='lw', cov_norm='l2', Lambda = 0.0001, component_num = 20)
                fcsdsp = fcsdsp.fit(trainset_o, trainset_label)  
                metric4.append(fcsdsp.score(testset_o, testset_label, cmp_num = idx))
                print(metric4[-1])
                
                jssdcpm_ds = JSSDCPM_dymatic_shrinkage(p = 2, shrinkage_type='ora', cov_norm='lp', component_num=idx)
                jssdcpm_ds = jssdcpm_ds.fit(trainset_o, trainset_label)  
                metric5.append(jssdcpm_ds.score(testset_o, testset_label, cmp_num = idx))
                print(metric3[-1])
                k = 1
                

    ave_metric1, ave_metric2, ave_metric3, ave_metric4, ave_metric5 = 0,0,0,0,0
    Metric1 = metric1.copy()
    Metric2 = metric2.copy()
    Metric3 = metric3.copy()
    Metric4 = metric4.copy()
    Metric5 = metric5.copy()
    
    idx = 1
    repeat_num = 10
    for i in range(repeat_num ):
        ave_metric1 += Metric1.pop()[idx]
        ave_metric2 += Metric2.pop()[idx]
        ave_metric3 += Metric3.pop()[idx]
        ave_metric4 += Metric4.pop()[idx]
        ave_metric5 += Metric5.pop()[idx]
        
        
    print(ave_metric1/repeat_num )
    print(ave_metric2/repeat_num )
    print(ave_metric3/repeat_num )
    print(ave_metric4/repeat_num )
    print(ave_metric5/repeat_num )
# %%
if __name__ == "__main__":
    idx = 3
    model1 = train_DCPM_noise_matrix(trainset[0:260,...], trainset_label,cmp_num = 20)
    metric1 = test_DCPM(testset[100:260,...], testset_label, model1, comp_num = idx)
    print(metric1)
    model2 = train_DCPM(trainset[0:260,...], trainset_label, 20)
    metric2 = test_DCPM(testset[0:260,...], testset_label, model2, comp_num = idx)
    print(metric2)

    num = 1
    plt.plot(-model1['filter'][:,num])
    plt.plot(model2['filter'][:,num])
    plt.show()

# %%
if __name__ == "__main__":
    jssdcpm = JSSDCPM(p = 2, shrinkage_type='ora', cov_norm='lp', component_num=20)
    jssdcpm = jssdcpm.fit(np.swapaxes(trainset[100:260,...], 0, 1), trainset_label)  
    a = jssdcpm.score(np.swapaxes(testset[100:260,...], 0, 1), testset_label, cmp_num = 3)
    print(a)
    jssdcpm_regu = JSSDCPM_regu(p = 2, alpha = 0, cov_norm='lp', component_num=20)
    jssdcpm_regu = jssdcpm_regu.fit(np.swapaxes(trainset[100:260,...], 0, 1), trainset_label)
    c = jssdcpm_regu.score(np.swapaxes(testset[100:260,...], 0, 1), testset_label, cmp_num = 3)
    print(c)

    model_JSSDCPM = train_JSSDCPM(np.swapaxes(trainset[100:260,...], 0, 1), 
                                    trainset_label,  
                                    component_num = 20, 
                                    p = 2, 
                                    shrinkage = True)
    a1 = test_JSSDCPM(testset[100:260,...], testset_label, model_JSSDCPM, comp_num = 3)
    print(a1)


    model_jssdcpm = dict()
    model_jssdcpm['filter'] = jssdcpm.filter
    model_jssdcpm['template of target'] = jssdcpm.target_tmp
    model_jssdcpm['template of nontarget'] = jssdcpm.nontarget_tmp
    model_jssdcpm['loss function'] = jssdcpm.loss_fun
    model_jssdcpm['shrinkage cofficient'] = jssdcpm.shrinkage_coff    
    a2 = test_JSSDCPM(testset[100:260,...], testset_label, model_jssdcpm, comp_num = 5)
    print(a2)
    k = 1
    #%%
    plt.figure()
    plt.plot(jssdcpm.filter[:,0])
    plt.plot(model_JSSDCPM['filter'][:,0])
    plt.figure()
    plt.plot(jssdcpm.filter[:,1])
    plt.plot(model_JSSDCPM['filter'][:,1])
    plt.figure()
    plt.plot(jssdcpm.filter[:,2])
    plt.plot(model_JSSDCPM['filter'][:,2])
    print(jssdcpm.shrinkage_coff)
    print(model_JSSDCPM['shrinkage cofficient'])
    k = 1
    
    # skgdcpm = SKGDCPM(p = 2, shrinkage_type = 'ora', component_num = 3)
    # skgdcpm = skgdcpm.fit(trainset, trainset_label)
    # b = skgdcpm.score(testset, testset_label, cmp_num = 3)
    # print(b)
    # model_skgdcpm = dict()
    # model_skgdcpm['filter'] = skgdcpm.filter
    # model_skgdcpm['template of target'] = skgdcpm.target_tmp
    # model_skgdcpm['template of nontarget'] = skgdcpm.nontarget_tmp
    # model_skgdcpm['loss function'] = skgdcpm.loss_fun
    # model_skgdcpm['shrinkage cofficient'] = skgdcpm.shrinkage_coff 
    # model_skgdcpm['p'] = skgdcpm.p
        
    # model_SKGDCPM = train_SKGDCPM(trainset, trainset_label, p = 2, component_num = 3,shrinkage = True)
    # b1 = test_SKGDCPM(testset, testset_label, model_SKGDCPM)
    # print(b1)
    # b2 = test_SKGDCPM(testset, testset_label, model_skgdcpm)
    # print(b2)
    
    #%%
    # plt.figure()
    # plt.plot(skgdcpm.filter[:,0])
    # plt.plot(model_SKGDCPM['filter'][:,0])
    # plt.figure()
    # plt.plot(skgdcpm.filter[:,1])
    # plt.plot(model_SKGDCPM['filter'][:,1])
    # plt.figure()
    # plt.plot(skgdcpm.filter[:,2])
    # plt.plot(model_SKGDCPM['filter'][:,2])
    # print(skgdcpm.shrinkage_coff)
    # print(model_SKGDCPM['shrinkage cofficient'])
    # k = 1
                

# %%
