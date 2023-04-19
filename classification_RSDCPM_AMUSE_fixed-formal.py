#%%
import copy
import scipy
from scipy import stats
import numpy as np
import cvxpy as cp
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from test_three_type_of_shrinkage_method import shrinkage_method
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#%%
def cal_score(testset_label, predict_label, criterion, TPlabel = 1, TNlabel = 0):
    """calcute different metrics of classification algorithm

    Args:
        testset_label (ndarry): true label of testdata
        predict_label (ndarry): predictive label of testdata
        criterion (ndarry): the classification criterion
        label_positive (int, optional): true positive label. Defaults to 1.
        label_negative (int, optional): false negative label. Defaults to 0.

    Returns:
        accuracy, balanced_accuracy, AUC, F1 score, TPR, TNR
    """
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
    """calcute the AUC of result

    Args:
        score (ndarray): the classification criterion
        target( ndarray): true label of testdata
        label_positive (int, optional): true positive label. Defaults to 1.
        label_negative (int, optional): false negative label. Defaults to 0.

    Returns:
        auc: the result of auc
    """
    
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

def transdataform(dataset):
    """Transform datatset from trial*chan*time to (time*chan)*tiral_n,
        and the integrity of the time vector is guaranteed

    Args:
        dataset (numpy.ndarray): trial*chan*time

    Returns:
        catted_data_at_time_dimension: (time*chan)*tiral_n
    """        
    tiral_n, chan_n, time_l = dataset.shape
    catted_data_at_time_dimension = np.reshape(dataset, (tiral_n, time_l * chan_n) , order = "C")
    
    return catted_data_at_time_dimension.T

def ave_downsample(dataset, decimate: int = 4):
    """_summary_

    Args:
        dataset (numpy.ndarray): trial*channel*time
        decimate (int, optional): decimate number. Defaults to 4.

    Returns:
        _type_: _description_
    """
    tiral_n, chan_n, time_l = dataset.shape
    win_num = np.floor(time_l/decimate).astype('int')
    
    if np.mod(time_l, decimate) == 0:
        new_dataset = np.zeros((tiral_n, chan_n, win_num))
        for i in range(win_num):
            slice_idx = slice(i, (i+1)*decimate)
            new_dataset[..., i] = dataset[..., slice_idx].mean(axis = 2)
    else:
        new_dataset = np.zeros((tiral_n, chan_n, win_num))
        for i in range(win_num):
            slice_idx = slice(i, (i+1)*decimate)
            new_dataset[..., i] = dataset[..., slice_idx].mean(axis = 2)
        
    return new_dataset

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
            
            
        # statistic classification accuracy
        self.predict_label = (np.sign(self.criterion) + 1) / 2
        return self.predict_label, self.criterion
    
    def cal_score(self, testset_label, predict_label, criterion, TPlabel = 1, TNlabel = 0):
        """calcute different metrics of classification algorithm

        Args:
            testset_label (ndarry): true label of testdata
            predict_label (ndarry): predictive label of testdata
            criterion (ndarry): the classification criterion
            label_positive (int, optional): true positive label. Defaults to 1.
            label_negative (int, optional): false negative label. Defaults to 0.

        Returns:
            accuracy, balanced_accuracy, AUC, F1 score, TPR, TNR
        """        
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
  
class DACIE(JSSDCPM):

    def __init__(
        self, 
        p = 1, 
        shrinkage_type = 'oas', 
        component_num =2, 
        tol=10**(-18), 
        max_iter = 40,
        interval_coffe = 0.3
    ):
        super().__init__(
            p = p, 
            shrinkage_type = shrinkage_type, 
            component_num = component_num, 
            tol = tol, 
            max_iter = max_iter
        )
        self.interval_coffe = interval_coffe
    
    def _get_ERP_interval_model(self, trainset, trainset_label):
        """get the ERP interval model through 
            Mean interval estimation

        Args:
            trainset (_type_): _description_
            trainset_label (_type_): _description_

        Returns:
            _type_: _description_
        """
        chan_num, time_len, trial_num = trainset.shape
        target_model = np.zeros((chan_num, time_len, 2))  # mid/upper/under
        ntarget_model = np.zeros((chan_num, time_len, 2)) # mid/upper/under
        
        target_trial = trainset[:, :, trainset_label == 1]
        ntarget_trial = trainset[:, :, trainset_label == 0]
        target_model[..., 0] = target_trial.mean(axis = 2)
        ntarget_model[..., 0] = ntarget_trial.mean(axis = 2)

        for i in range(target_trial.shape[0]):
            for j in range(target_trial.shape[1]):
                
                arr = target_trial[i,j,:]
                alpha = 0.05                  # significance level = 5%
                df = arr.size - 1                  # degress of freedom = 20
                t = stats.t.ppf(1 - alpha/2, df)   # t-critical value for 95% CI = 2.093
                s = np.std(arr, ddof=1)            # sample standard deviation = 2.502
                n = arr.size
                target_upper = np.mean(arr) + (t * s / np.sqrt(n))
                target_lower = np.mean(arr) - (t * s / np.sqrt(n))
                target_model[..., 1] =  (target_upper - target_lower)/2
                
                arr2 = ntarget_trial[i,j,:]
                alpha = 0.05                    # significance level = 5%
                df = arr2.size - 1                  # degress of freedom = 20
                t = stats.t.ppf(1 - alpha/2, df)   # t-critical value for 95% CI = 2.093
                s = np.std(arr2, ddof=1)            # sample standard deviation = 2.502
                n = arr2.size
                ntarget_upper = np.mean(arr2) + (t * s / np.sqrt(n))
                ntarget_lower = np.mean(arr2) - (t * s / np.sqrt(n))
                ntarget_model[..., 1] = (ntarget_upper - ntarget_lower)/2

        return target_model, ntarget_model
       
    def _fuzzy_set_square_distance(self, interval_model1, interval_model2):
        """Calcute the square of distance defined on fuzzy set

        Args:
            interval_model1 (ndarray): shape: [mid_matrix, upper_matrix, lower_matrix]
            interval_model2 (ndarray): shape: [mid_matrix, upper_matrix, lower_matrix]
            
        return:
            cov_matrix (ndarray)
        """
    
        if len(interval_model1.shape) == 3:
            interval_model1_for_cal = interval_model1
        else:
            chan_num, time_len = interval_model1.shape
            interval_model1_for_cal = np.zeros((chan_num,time_len, 2))
            interval_model1_for_cal[..., 0] = interval_model1

        if len(interval_model2.shape) == 3:
            interval_model2_for_cal = interval_model2
        else:
            chan_num, time_len = interval_model2.shape
            interval_model2_for_cal = np.zeros((chan_num,time_len, 2))
            interval_model2_for_cal[..., 0] = interval_model2
        
        m_diff = interval_model1_for_cal[..., 0] - interval_model2_for_cal[..., 0]
        r_diff = interval_model1_for_cal[..., 1] - interval_model2_for_cal[..., 1]
        square_distance = m_diff @  m_diff.T + self.interval_coffe * r_diff @  r_diff.T
        
        return square_distance
           
    def _calculate_tROI(self, trainset, trainset_label, cmp_num):
        """calcute time region of interest,ie. Fisher score 
            based on the fuzzy set distance

        Args:
            dataset (ndarray): shape(chan_num, time_len, trial_num)
            dataset_label (ndarray): shape(trial_num,)
            cmp_num (float): component_num for transform
        """
        ftrainset = self.transform(trainset, cmp_num = cmp_num, atype = "S")
        self.ftarget_model, self.fntarget_model = self._get_ERP_interval_model(ftrainset, trainset_label)
        ftarget_model = self.ftarget_model
        fntarget_model = self.fntarget_model
        chan_num, time_len, trial_num = ftrainset.shape
        # get class trial
        ftarget = ftrainset[..., trainset_label.squeeze() == 1]
        fnontarget = ftrainset[..., trainset_label.squeeze() == 0]
        b = np.zeros((time_len,))
        for i in range(time_len):
            Db = self._fuzzy_set_square_distance(ftarget_model[:, i, None, :], fntarget_model[:, i, None, :])
            Dw1 = 0
            for j in range(ftarget.shape[2]):
                Dw1 += self._fuzzy_set_square_distance(ftarget[:, i, None, j], ftarget_model[:, i, None, :])
            Dw0 = 0
            for j in range(fnontarget.shape[2]):
                Dw0 += self._fuzzy_set_square_distance(fnontarget[:, i, None, j], fntarget_model[:, i, None, :])
            
            numerator = np.linalg.norm(Db, ord = 'fro')
            denominator = np.linalg.norm(Dw0 + Dw1, ord = 'fro')
            b[i] = numerator/denominator

        self.tROI = b
        return self.tROI , ftrainset
        
    def _discriminant_eigenvalue(self, single_trial, cmp_num):
        """calculate the discriminate eigenvalue between
            filtered single trial data and the middle of interval
            model

        Args:
            single_trial (ndarray): shape[n_cmp, time_len]
            cmp_num (ndarray): component number used for test

        Returns:
            feature (ndarray): shape[n_cmp, ]  the discriminate 
                                eigenvalue of the inpute trial
        """
        S1m = self.st_target_model[0:cmp_num, :, 0]
        R1 = (single_trial - S1m) @ (single_trial - S1m).T
        eigenvalue1 , _ = np.linalg.eig(R1)
        idx = np.argsort(eigenvalue1)
        eigenvalue1 =eigenvalue1[idx]
        
        S0m = self.st_ntarget_model[0:cmp_num, :, 0]
        R0 = (single_trial - S0m) @ (single_trial - S0m).T
        eigenvalue0 , _ = np.linalg.eig(R0)
        idx = np.argsort(eigenvalue0)
        eigenvalue0 =eigenvalue0[idx]
        
        feature = np.real(eigenvalue1)/(np.real(eigenvalue0)+10**(-25))
        return feature
 
    def fit(self, trainset, trainset_label):
        """fit the model by transform Rayleigh optimization problem to 
            quadric form (same with paper) 

        Args:
            trainset (ndarray): shape[n_chan, n_time, n_trial]
            trainset_label (ndarray): shape[n_chan, n_time, n_trial]

        Returns:
            self
        """        
        cmp_num = self.cmp_num
        p = self.p
        shrinkage = self.shrinkage
        max_iter = self.max_iter
        stop_criterion = self.tol
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        # data centralization
        for i in range(trainset.shape[2]):
            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
            
        # get class trial
        target = train_trial[..., trainset_label.squeeze() == 1]
        nontarget = train_trial[..., trainset_label.squeeze() == 0]
        
        # get interval model of ERP
        self.target_model, self.ntarget_model = self._get_ERP_interval_model(train_trial, trainset_label)
        
        # calcute  between-class divergence matrix
        Db = self._fuzzy_set_square_distance(self.target_model, self.ntarget_model)/time_len

        # calcute intra-class divergence matrix
        cov_all2 = np.zeros([chan_num, chan_num, trial_num])
        cov_all3 = np.zeros([chan_num, chan_num, trial_num])
        for n in range(target.shape[2]):
            cov_all2[..., n] = self._fuzzy_set_square_distance(target[..., n].squeeze(), self.target_model)
                                
        cov_0 = np.mean(cov_all2, axis = 2)
        for n in range(nontarget.shape[2]):
            cov_all3[..., n] = self._fuzzy_set_square_distance(nontarget[..., n].squeeze(), self.ntarget_model)
            
        cov_1 = np.mean(cov_all3, axis = 2)
        Dw = (cov_0 + cov_1)/(2*time_len)
        
        # calcute the shrinkage coefficient and shrinkage Sw
        if shrinkage is None:
            P = Dw.shape[1]
            F = np.trace(Dw)/P
            alpha = 0
        else:

            P = Dw.shape[1]
            F = np.trace(Dw)/P
            Tar = F * (np.eye(Dw.shape[0])) 
            shrink = shrinkage_method(trainset, Dw, Tar)
            match shrinkage:
                case "oas" :
                    alpha, _ = shrink.oracle()
                case "lw" :
                    alpha, _ = shrink.ledoit_wolf()
                case "rblw":
                    alpha, _ = shrink.rao_blackwell_LW()
                case "ss":
                    alpha, _ = shrink.schafe_strimmer()
    
        A = np.diag(np.ones((chan_num, )))
        mu = 10**(-9)
        xx = np.linalg.norm(A, ord = 2, axis = 0)** (2-p)
        xx[xx == np.inf] = 10**20
        G = np.diag(1/(xx + mu) ) 
        
        for i in range(max_iter):
            
            eig_value, eig_vector = np.linalg.eig(Dw)
            a = eig_vector @ np.diag(eig_value) @ eig_vector.T
            eig_value_diag = np.diag(eig_value)
            eig_value_matrix = np.diag(1/eig_value**(1/2))
            U = np.real(eig_vector)
            
            aim = eig_value_matrix @ U.T @ ((1 - alpha)*Db - alpha *F* G) @ U @ eig_value_matrix
            # solve the optimizatino problem
            svd_value , right_vector = np.linalg.eig(aim)
            denote_idx = np.argsort(svd_value)
            denote_idx = np.flip(denote_idx)
            sorted_V = svd_value[denote_idx]
            sorted_W = right_vector[:,denote_idx]
            Ar = np.real(sorted_W[:,0:cmp_num].T)
            # Ar = right_vector.T
            
            A = Ar @ eig_value_matrix @ U.T
            xx = np.linalg.norm(A, ord = 2, axis = 0)** (2-p)
            xx[xx == np.inf] = 10**20
            G = np.diag(1/(xx + mu) ) 
            
            if i >= 1:
                Jt = np.mean(np.linalg.norm(np.linalg.norm(Ar @ aim @ Ar.T, axis = 0), axis = 0))
                Jt_1 = np.mean(np.linalg.norm(np.linalg.norm(Ar_old @ aim_old @ Ar_old.T, axis = 0), axis = 0))
                # print( np.abs(Jt_1 - Jt))
                if np.abs(Jt_1 - Jt) < stop_criterion:
                    break
            
            aim_old = aim
            Ar_old = Ar
            
        # save DCPM model
        self.filter = np.real(A.T)
        self.target_tmp = self.target_model[..., 0]
        self.nontarget_tmp = self.ntarget_model[..., 0]
        
        # calcute time region of interest
        self._calculate_tROI(train_trial, trainset_label, cmp_num = cmp_num)
        # calcute discriminant eigenvalue of trainset
        self.st_trainset = self.transform(train_trial, cmp_num = cmp_num, atype = "S-T")
        self.trainset_label = trainset_label
        self.st_target_model, self.st_ntarget_model = self._get_ERP_interval_model(self.st_trainset, trainset_label)
        
        return self
    
    def fit2(self, trainset, trainset_label):
        """fit the model by jssdcpm method

        Args:
            trainset (ndarray): shape[n_chan, n_time, n_trial]
            trainset_label (ndarray): shape[n_chan, n_time, n_trial]

        Returns:
            self
        """
        cmp_num = self.cmp_num # component num for fit
        p = self.p
        shrinkage = self.shrinkage
        max_iter = self.max_iter
        stop_criterion = self.tol
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        # data centralization
        for i in range(trainset.shape[2]):
            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
            
        # get class trial
        target = train_trial[..., trainset_label.squeeze() == 1]
        nontarget = train_trial[..., trainset_label.squeeze() == 0]
        
        # get interval model of ERP
        self.target_model, self.ntarget_model = self._get_ERP_interval_model(train_trial, trainset_label)
        
        # calcute  between-class divergence matrix
        Db = self._fuzzy_set_square_distance(self.target_model, self.ntarget_model)/time_len

        # calcute intra-class divergence matrix
        cov_all2 = np.zeros([chan_num, chan_num, trial_num])
        cov_all3 = np.zeros([chan_num, chan_num, trial_num])
        for n in range(target.shape[2]):
            cov_all2[..., n] = self._fuzzy_set_square_distance(target[..., n].squeeze(), self.target_model)
                                
        cov_0 = np.mean(cov_all2, axis = 2)
        for n in range(nontarget.shape[2]):
            cov_all3[..., n] = self._fuzzy_set_square_distance(nontarget[..., n].squeeze(), self.ntarget_model)
            
        cov_1 = np.mean(cov_all3, axis = 2)
        Dw = (cov_0 + cov_1)/(2 * time_len)
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        if shrinkage is None:
            P = Dw.shape[1]
            F = np.trace(Dw)/P
            alpha = 0
        else:

            P = Dw.shape[1]
            F = np.trace(Dw)/P
            Tar = F * (np.eye(Dw.shape[0])) 
            shrink = shrinkage_method(trainset, Dw, Tar)
            match shrinkage:
                case "oas" :
                    alpha, _ = shrink.oracle()
                case "lw" :
                    alpha, _ = shrink.ledoit_wolf()
                case "rblw":
                    alpha, _ = shrink.rao_blackwell_LW()
                case "ss":
                    alpha, _ = shrink.schafe_strimmer()
    
        A = np.diag(np.ones((chan_num, )))
        mu = 10**(-9)
        xx = np.linalg.norm(A, ord = 2, axis = 0)** (2-p)
        xx[xx == np.inf] = 10**20
        G = np.diag(1/(xx + mu) ) 
        
        criterion_register = list()
        for i in range(max_iter):
            
            # get aim matrix
            aim = np.linalg.pinv((1 - alpha) * Dw + alpha * F * G) @ Db
            # solve the optimizatino problem
            svd_value , right_vector = scipy.linalg.eigh(Db, (1 - alpha) * Dw + alpha * F * G)
            denote_idx = np.argsort(svd_value)
            denote_idx = np.flip(denote_idx)
            sorted_V = svd_value[denote_idx]
            sorted_W = right_vector[:,denote_idx]
            A = np.real(sorted_W[:,0:cmp_num].T)
            # calcute normalization matrix
            xx = np.linalg.norm(A, ord = 2, axis = 0)** (2-p)
            xx[xx == np.inf] = 10**20
            G = np.diag(1/(xx + mu) ) 
            
            if i >= 1:
                Jt_1_Jt = np.trace(A @ G @ A.T)-np.trace(A_old @ G_old @ A_old.T)
                criterion_register.append(np.abs(Jt_1_Jt))
                # print( np.abs(Jt_1_Jt))
                if np.abs(Jt_1_Jt) < stop_criterion:
                    break
                
            A_old = A
            G_old = G

        # save DCPM model
        self.filter = np.real(A.T)
        self.target_tmp = self.target_model[..., 0]
        self.nontarget_tmp = self.ntarget_model[..., 0]
        self.loss_fun = criterion_register
        
        # calcute time region of interest
        self._calculate_tROI(train_trial, trainset_label, cmp_num = cmp_num)
        # calcute discriminant eigenvalue of trainset
        self.st_trainset = self.transform(train_trial, cmp_num = cmp_num, atype = "S-T")
        self.trainset_label = trainset_label
        self.st_target_model, self.st_ntarget_model = self._get_ERP_interval_model(self.st_trainset, trainset_label)

        return self
    
    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)

    def transform(self, dataset, cmp_num, atype = "S"):
        """projecting inpute dataset to source space 

        Args:
            dataset (ndarray): shape[n_chan, n_time, n_trial]
            cmp_num (int): component number used for test
            atype (str, "S" , "S-T"): the type of projection, "S" projecting
                            dataset to source space, "S-T" executing the operation
                            besides of "S", this option using tROI as the weight of 
                            each time point. Defaults to "S".

        Returns:
            filtered_dataset (ndarray): shape[cmp_num, n_time, n_trial]
        """
        if atype == "S":
            return super().transform(dataset, cmp_num)
        
        elif atype == "S-T":
            B = np.diag(self.tROI)
            _, time_len, trial_num = dataset.shape
            filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
            W = self.filter[:,0:cmp_num]
            for i in range(trial_num):
                filtered_dataset[..., i] = W.T @ dataset[..., i] @ B
            return filtered_dataset

    def fit_transform(self, trainset, trainset_label, cmp_num):
        return super().fit_transform(trainset, trainset_label, cmp_num)
    
    def predict(self, testset, cmp_num, atype = 'pattern matching'):
        """predict the label of testset using "pattern matching" method
            or "discriminant eigenvalue" method

        Args:
            testset (ndarray): shape[n_chan, n_time, n_trial]
            cmp_num (int): component number of spatial filter used for test
            atype (str, 'pattern matching'or'discriminant eigenvalue'): 
                            Defaults to 'pattern matching'.

        Returns:
            predict_label (ndarray): shape[n_trial, ]
            criterion (ndarray): shape[n_trial, ]
            
        """
        if atype == 'pattern matching':
            return super().predict(testset, cmp_num)
        elif atype == 'discriminant eigenvalue':
            # centralization
            location = np.mean(testset, axis = 1, keepdims = True)
            testset = testset - location
            # extract testset feature
            trial_num = testset.shape[2]
            st_testset = self.transform(testset, cmp_num = cmp_num, atype = 'S-T')
            test_eig_feature = np.zeros((testset.shape[2], cmp_num))
            for i in range(testset.shape[2]):
                test_eig_feature[i,:] = self._discriminant_eigenvalue(st_testset[:,:,i], cmp_num)
            # extract trainset feature
            st_trainset = self.st_trainset
            train_eig_feature = np.zeros((st_trainset.shape[2], cmp_num))
            for i in range(st_trainset.shape[2]):
                train_eig_feature[i,:] = self._discriminant_eigenvalue(st_trainset[0:cmp_num,...][...,i], cmp_num)
            # preprocessing
            scaler =  MinMaxScaler()
            ztrain_eig_feature = scaler.fit_transform(train_eig_feature[:,:])
            ztest_eig_feature = scaler.transform(test_eig_feature[:,:])
            # classification
            self.clf = SVC(
                C= 4.0, 
                kernel="linear", 
                degree= 3 , 
                gamma='scale', 
                coef0= 0.1, 
                shrinking=True, 
                probability=False, 
                tol=0.0001, 
                cache_size=1000, 
                class_weight= 'balanced', 
                verbose=False, 
                max_iter=1000, 
                decision_function_shape='ovo', 
                break_ties=False, 
                random_state=79)

            self.clf.fit(ztrain_eig_feature, self.trainset_label)
            self.criterion = self.clf.decision_function(ztest_eig_feature)
            self.predict_label = self.clf.predict(ztest_eig_feature)
            # self.lda = LinearDiscriminantAnalysis(
            #     solver= 'eigen', 
            #     shrinkage='auto', 
            #     priors=None, 
            #     n_components=None, 
            #     store_covariance=False, 
            #     tol=0.0001, 
            #     covariance_estimator=None
            # )
            # self.lda.fit(ztrain_eig_feature[:,0:], self.trainset_label)
            # self.predict_label = self.lda.predict(ztrain_eig_feature[:,0:])
            # self.criterion  = self.lda.decision_function(ztest_eig_feature[:,0:])
            return self.predict_label, self.criterion  
    
    def score(self, testset, testset_label, cmp_num, atype = 'pattern matching', TPlabel = 1, TNlabel = 0):
          
        if atype == 'pattern matching':
            return super().score(testset, testset_label, cmp_num, TPlabel = 1, TNlabel = 0) 
        elif atype == 'discriminant eigenvalue':
            predict_label, criterion = self.predict(testset, cmp_num = cmp_num, atype = 'discriminant eigenvalue')
            return super().cal_score(testset_label, predict_label, criterion, TPlabel, TNlabel) 

class MTS_estimator1():
    """Calculate the multi-target shrinkage estimation of sample mean
    """
    def __init__(self, cluster_n=0):
        self.cluster_n = cluster_n
    
    def _transdataform(self, dataset):
        """Transform datatset from trial*chan*time to (time*chan)*tiral_n,
            and the integrity of the time vector is guaranteed

        Args:
            dataset (numpy.ndarray): trial*chan*time

        Returns:
            catted_data_at_time_dimension: (time*chan)*tiral_n
        """
        tiral_n, chan_n, time_l = dataset.shape
        catted_data_at_time_dimension = np.reshape(dataset, (tiral_n, time_l * chan_n) , order = "C")
        return catted_data_at_time_dimension.T
    
    def _transdataform2(self, dataset):
        """Transform datatset from trial*channel*time to (time*trial)*channels,
            and the integrity of the time vector is guaranteed

        Args:
            dataset (numpy.ndarray): trial*chan*time

        Returns:
            catted_data_at_time_dimension: (time*trial)*channels
        """
        tiral_n, chan_n, time_l = dataset.shape
        dataset2 = np.transpose(dataset, (1, 2, 0))
        catted_data_at_time_dimension = np.reshape(dataset2, (chan_n, time_l * tiral_n) , order = "F")
        return catted_data_at_time_dimension.T

    def clustering(self, dataset_2d:np.ndarray, data_label):
        """if there are no subclass labels for dataset, then the clustring
        method is carried out one the rest-sate data to generate subclass
        labels
        Args:
            dataset_2d (numpy.ndarray): the rest-state data
            data_label (numpy.ndarray): class label of rest-state data

        Returns:
            self: 'self.label' contain the result of clustring analysis
        """
        self.dataset = dataset_2d
        self.data_label = data_label
        cluster_n = self.cluster_n
        # get the shape of dataset
        self.trial_n, self.chan_n, self.time_l = dataset_2d.shape
        trial_n, chan_n, time_l = self.trial_n, self.chan_n, self.time_l
        # transform the dimension of dataset from 2d to 1d
        dataset_1d_chan_mean = self._transdataform(self.dataset)
        dataset_1d_chan_mean = np.std(self.dataset, axis = 1)
        # clustering the data and get the label fo each eeg sample
        clf =  SpectralClustering(n_clusters = cluster_n, assign_labels='cluster_qr', affinity='nearest_neighbors', n_neighbors = 10, random_state=0)
        clf.fit(dataset_1d_chan_mean) # affinity='nearest_neighbors', n_neighbors = 10, 
        self.label = clf.labels_
        return self
        
    def fit(self, dataset, dataset_label, whitening = False):
        """Solve the quadratic optimization problem corresponding to
        the multi-target shringk estimation of sample mean

        Args:
            dataset (numpy.ndarray): trial*channel*time
            dataset_label (_numpy.ndarray): label of dataset
            whitening (bool, optional): weather whitening the dataset before shrinkage. Defaults to False.

        Returns:
            self
        Bartz, D., Höhne, J., Müller, K.-R., 2014. Multi-Target Shrinkage, Submitted—Available on arXiv.
        """
        self.type_data = dataset
        # whitening the data
        if whitening is True:
            whitening_transformed_data = self._transdataform2(self.type_data)
            pca = PCA(n_components = int(dataset.shape[1]/2), whiten = True)
            pca.fit(whitening_transformed_data)
            whitening_filter = pca.components_
            whitening_dataset = np.einsum('tcs, kc -> tks', self.type_data, whitening_filter)
            self.type_data = (whitening_dataset )
        
        dataset_2d = self.type_data
        # get the shape of dataset
        self.trial_n, self.chan_n, self.time_l = dataset_2d.shape
        trial_n, chan_n, time_l = self.trial_n, self.chan_n, self.time_l
        self.type_label = dataset_label
        label = self.type_label
        self.label_type = np.unique(label)
        cluster_n = self.label_type.size
        self.cluster_n = cluster_n
        # get the base parameter for Quadratic Programming(QP) problem
        data_2d_list = np.zeros((chan_n, time_l, cluster_n))
        subtrial_n_list = np.zeros((cluster_n))
        b_square = np.zeros((cluster_n, chan_n, time_l))
        b = np.zeros((cluster_n ,))
        
        for key_idx, keys in enumerate(self.label_type):
            
            keys = int(keys)
            idx = np.argwhere(label == keys).squeeze()
            sub_trial_n = idx.size
            subtrial_n_list[key_idx] = 1/sub_trial_n
            data_2d_list[...,key_idx] = dataset_2d[idx,...].mean(axis = 0)
            Ck = dataset_2d[idx,...] - data_2d_list[...,key_idx]
            
            for nq in range(sub_trial_n):
                b_square[key_idx]+= (1/(sub_trial_n*(sub_trial_n-1))) * Ck[nq,...]**2
                
            b[key_idx] = b_square[key_idx].sum()
           
        # struct the QP through the cvxpy lib and solve the problem for each subclass
        self.value = np.zeros((cluster_n, cluster_n))
        for k in range(cluster_n):
            
            A = np.zeros((cluster_n, cluster_n))
            for i in range(cluster_n):
                for j in range(cluster_n):
                    CC = (data_2d_list[..., i] - data_2d_list[..., k]) * (data_2d_list[..., j] - data_2d_list[..., k])
                    A[i,j] = CC.sum()
                                
            # define Variable of QP problem
            pars = cp.Variable((cluster_n), nonneg = True)
            # define the Optimization Goals 
            obj = cp.Minimize(cp.quad_form(pars, A)/2 - pars @ b)
            # define the constrain condition          
            other_var = cp.multiply(pars, subtrial_n_list)
            sum_par = 0
            for par_idx in range(cluster_n):
                if par_idx!= k:
                    sum_par += pars[par_idx]
            constrain = [pars[k] == 0, cp.multiply(pars, subtrial_n_list) <= (1 - cp.sum(pars) + pars[k])*subtrial_n_list[k]]
            # struct optimization problem
            prob = cp.Problem(obj, constrain)
            # solve the problem
            try:
            #    prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-6),max_iters = 30000)
               prob.solve(solver = 'OSQP', verbose=False, eps_rel = 10**(-7), max_iter = 30000)
               
            except:
                try:
                    # prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-5),max_iters = 30000)
                    prob.solve(solver = 'OSQP', verbose=False, eps_rel = 10**(-4), max_iter = 30000)
                except:
                    try:
                        # prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-3),max_iters = 30000)
                        prob.solve(solver = 'OSQP', verbose=False, eps_rel = 10**(-3), max_iter = 30000)
                    except:
                        try:
                            # prob.solve(solver = 'ECOS', verbose=True, abstol = 10**(-2),max_iters = 30000)
                            prob.solve(solver = 'OSQP', verbose=False, eps_rel = 10**(-2), max_iter = 30000)
                        except:
                            # prob.solve(solver = 'ECOS', verbose=True, abstol = 10**(-1),max_iters = 30000)
                            prob.solve(solver = 'OSQP', verbose=True, eps_rel = 10**(-1), max_iter = 30000)
            self.value[k, :] = pars.value
            
        value_triu = np.triu(self.value)
        value_tril = np.tril(self.value)
        tri_matrix = (value_triu + value_tril.T)/2
        self.value = np.zeros_like(self.value)
        self.value = tri_matrix + tri_matrix.T
        
        value = self.value.copy()
        c_idx = [cidx for cidx in range(cluster_n)]
        for i in range(cluster_n):
            for j in range(cluster_n):
                if i == j:
                    value[i, j] = 1 - value[i, c_idx!=j].sum()
        sn.heatmap(value)
        plt.show()
        return self
    
    def get_shrinkage_mean(self, dataset = None):
        """get the shrinkage estimated sample mean of given dataset
        Args:
            dataset (numpy.ndarray, optional): subclass*channel*time 
                            the dataset include sample mean of different subclass,
                            all subclasse belong to same class. Defaults to None.
        Returns:crespond
            sk_means_2d_list: the shrinkage estimation of sample mean corresponding
                                to different subclass
            data_2d_list: sample mean corresponding
                                to different subclass
            value: subclass * subclass, shrinkage cofficient of each subclass 
        """
        if dataset is None:
            # get the base parameter for Quadratic Programming(QP) problem
            dataset_2d  = self.type_data 
            trial_n, chan_n, time_l = self.trial_n, self.chan_n, self.time_l
        else:
            # dataset_2d  = (dataset - dataset.mean(axis = 2, keepdims= True))
            dataset_2d  = dataset
            trial_n, chan_n, time_l = dataset.shape
            
        data_2d_list = np.zeros((chan_n, time_l, self.cluster_n))
        for key_idx, keys in enumerate(self.label_type):
            keys = int(keys)
            data_2d_list[..., key_idx] = dataset_2d[self.type_label == keys, ...].mean(axis = 0)
            
        # struct the QP through the cvxpy lib and solve the problem for each subclass
        sk_means_2d_list = np.zeros(( chan_n, time_l, self.cluster_n))
        value = self.value
        for k in range(self.cluster_n):
            register_mean_2d = copy.deepcopy(data_2d_list)
            
            for t in range(self.cluster_n):
                if t == k:
                    register_mean_2d[..., t] = 0 * register_mean_2d[..., t]
                register_mean_2d[..., t] = value[k, t] * register_mean_2d[..., t]
            sk_means_2d_list[..., k] = (1 - np.sum(value[k,:])) * data_2d_list[..., k] + register_mean_2d.sum(axis = 2)
                
        return sk_means_2d_list, data_2d_list, value

class MTS_estimator():
    """Calculate the multi-target shrinkage estimation of sample mean
    """   
    def __init__(self, cluster_n = 0):
        self.cluster_n = cluster_n
    
    def _transdataform(self, dataset):
        """Transform datatset from trial*chan*time to (time*chan)*tiral_n,
            and the integrity of the time vector is guaranteed

        Args:
            dataset (numpy.ndarray): trial*chan*time

        Returns:
            catted_data_at_time_dimension: (time*chan)*tiral_n
        """        
        tiral_n, chan_n, time_l = dataset.shape
        catted_data_at_time_dimension = np.reshape(dataset, (tiral_n, time_l * chan_n) , order = "C")
        return catted_data_at_time_dimension.T
    
    def _transdataform2(self, dataset):
        """Transform datatset from trial*channel*time to (time*trial)*channels,
            and the integrity of the time vector is guaranteed

        Args:
            dataset (numpy.ndarray): trial*chan*time

        Returns:
            catted_data_at_time_dimension: (time*trial)*channels
        """        
        tiral_n, chan_n, time_l = dataset.shape
        dataset2 = np.transpose(dataset, (1, 2, 0))
        catted_data_at_time_dimension = np.reshape(dataset2, (chan_n, time_l * tiral_n) , order = "F")
        return catted_data_at_time_dimension.T

    def clustering(self, dataset_2d:np.ndarray, data_label):
        
        self.dataset = dataset_2d
        # self.dataset = (dataset_2d - dataset_2d.mean(axis = 1, keepdims=True)) #/ np.std(dataset_2d, axis = 1, keepdims=True)
        self.data_label = data_label
        cluster_n = self.cluster_n
        # get the shape of dataset
        self.trial_n, self.chan_n, self.time_l = dataset_2d.shape
        trial_n, chan_n, time_l = self.trial_n, self.chan_n, self.time_l
        # transform the dimension of dataset from 2d to 1d
        dataset_1d_chan_mean = self._transdataform(self.dataset)
        dataset_1d_chan_mean = np.std(self.dataset, axis = 1)
        # clustering the data and get the label fo each eeg sample
        clf =  SpectralClustering(n_clusters = cluster_n, assign_labels='cluster_qr', affinity='nearest_neighbors', n_neighbors = 10, random_state=0)
        clf.fit(dataset_1d_chan_mean) # affinity='nearest_neighbors', n_neighbors = 10, 
        self.label = clf.labels_
        return self
        
    def fit(self, dataset, dataset_label, whitening = False):
        """Solve the quadratic optimization problem corresponding to
        the multi-target shringk estimation of sample mean, and the multiple quadratic
        optimization problems of each shrinkage coefficient was reformed to one
        single quadratic optimization problem.

        Args:
            dataset (numpy.ndarray): trial*channel*time
            dataset_label (_numpy.ndarray): label of dataset
            whitening (bool, optional): weather whitening the dataset before shrinkage. Defaults to False.

        Returns:
            self
            
        Bartz, D., Höhne, J., Müller, K.-R., 2014. Multi-Target Shrinkage, Submitted—Available on arXiv.
        """
        self.type_data = dataset
        # whitening the data
        if whitening is True:
            whitening_transformed_data = self._transdataform2(self.type_data)
            pca = PCA(n_components = int(dataset.shape[1]/2), whiten = True)
            pca.fit(whitening_transformed_data)
            whitening_filter = pca.components_
            whitening_dataset = np.einsum('tcs, kc -> tks', self.type_data, whitening_filter)
            self.type_data = (whitening_dataset )
        
        dataset_2d = self.type_data
        # get the shape of dataset
        self.trial_n, self.chan_n, self.time_l = dataset_2d.shape
        trial_n, chan_n, time_l = self.trial_n, self.chan_n, self.time_l
        self.type_label = dataset_label
        label = self.type_label
        self.label_type = np.unique(label)
        self.cluster_n = self.label_type.size
        cluster_n = self.cluster_n
        # get the base parameter for Quadratic Programming(QP) problem
        data_2d_list = np.zeros((chan_n, time_l, cluster_n))
        subtrial_n_list = np.zeros((cluster_n))
        n_all_sample = 0
        b_square = np.zeros((cluster_n, chan_n, time_l))
        b = np.zeros((cluster_n ,1))
        
        for key_idx, keys in enumerate(self.label_type):
            
            keys = int(keys)
            idx = np.argwhere(label == keys).squeeze()
            sub_trial_n = idx.size
            subtrial_n_list[key_idx] = 1/sub_trial_n
            n_all_sample += sub_trial_n
            data_2d_list[...,key_idx] = dataset_2d[idx,...].mean(axis = 0)
            Ck = dataset_2d[idx,...] - data_2d_list[...,key_idx]
            
            for nq in range(sub_trial_n):
                b_square[key_idx]+= (1/(sub_trial_n*(sub_trial_n-1))) * Ck[nq,...]**2
                
            b[key_idx] = b_square[key_idx].sum()
        b_list = list()
        subtrial_n_list_list = list()
        for i in range(cluster_n):
            b_list.append(b)
            subtrial_n_list_list.append(subtrial_n_list)
        b_vector = cp.vstack(b_list)
        subtrial_n_list_vector = np.hstack(subtrial_n_list_list)
        # struct the QP through the cvxpy lib and solve the problem for each subclass
        self.value = np.zeros((cluster_n, cluster_n))
        A = np.zeros((cluster_n*cluster_n, cluster_n*cluster_n))
        for k in range(cluster_n):
            
            for i in range(cluster_n):
                for j in range(cluster_n):
                    CC = (data_2d_list[..., i] - data_2d_list[..., k]) * (data_2d_list[..., j] - data_2d_list[..., k])
                    A[k*cluster_n + i, k*cluster_n + j] = CC.sum()

        # define Variable of QP problem
        pars = cp.Variable((cluster_n*cluster_n), nonneg = True)
        # define the Optimization Goals 
        obj = cp.Minimize(cp.quad_form(pars, A)/2 - pars @ b_vector)
        
        # define the constrain condition
        zero_s_idx = [i + i_name for i, i_name in enumerate(range(0, cluster_n*cluster_n, cluster_n))]
        segment_idx = [i for i in range(0, cluster_n*cluster_n+1, cluster_n)]
        pars_idx = np.zeros((cluster_n, cluster_n-1),dtype='int16')
        for i in range(cluster_n):
            count = 0
            for j in range(segment_idx[i], segment_idx[i+1]):
                if j != zero_s_idx[i]:
                    pars_idx[i, count] = j
                    count+= 1
        constrain_part1 = [pars[zero_s_idx] == 0]
        constrain_part2 = list()
        for k in range(cluster_n):
            idx_k = zero_s_idx[k]
            constrain_part2.append(cp.multiply(pars[pars_idx[k, :]], subtrial_n_list_vector[pars_idx[k, :]])<= (1-cp.sum(pars[pars_idx[k,:]]))*subtrial_n_list_vector[idx_k])
        constrain = constrain_part1 + constrain_part2
                 
        # struct optimization problem
        prob = cp.Problem(obj, constrain)
        # solve the problem
        try:
            prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-6),max_iters = 30000)
        except:
            try:
                prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-5),max_iters = 30000)
            except:
                try:
                    prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-3),max_iters = 30000)
                except:
                    prob.solve(solver = 'ECOS', verbose=False, abstol = 10**(-2),max_iters = 30000)
                    
        for i in range(cluster_n):
            for j in range(cluster_n):
                self.value[i, j] = pars.value[i*cluster_n+j]
        
            
        # value = self.value.copy()
        # c_idx = [cidx for cidx in range(cluster_n)]
        # for i in range(cluster_n):
        #     for j in range(cluster_n):
        #         if i == j:
        #             value[i, j] = 1 - value[i, c_idx!=j].sum()
        # sn.heatmap(value)
        # plt.show()
        
        return self
    
    def get_shrinkage_mean(self, dataset = None):
        """get the shrinkage estimated sample mean of given dataset
        Args:
            dataset (numpy.ndarray, optional): subclass*channel*time 
                            the dataset include sample mean of different subclass,
                            all subclasse belong to same class. Defaults to None.
        Returns:crespond
            sk_means_2d_list: the shrinkage estimation of sample mean corresponding
                                to different subclass
            data_2d_list: sample mean corresponding
                                to different subclass
            value: subclass * subclass, shrinkage cofficient of each subclass 
        """
        if dataset is None:
            # get the base parameter for Quadratic Programming(QP) problem
            dataset_2d  = self.type_data 
            trial_n, chan_n, time_l = self.trial_n, self.chan_n, self.time_l
        else:
            # dataset_2d  = (dataset - dataset.mean(axis = 2, keepdims= True))
            dataset_2d  = dataset
            
            trial_n, chan_n, time_l = dataset.shape
            
        data_2d_list = np.zeros((chan_n, time_l, self.cluster_n))
        for key_idx, keys in enumerate(self.label_type):
            keys = int(keys)
            data_2d_list[..., key_idx] = dataset_2d[self.type_label == keys, ...].mean(axis = 0)
            
        # struct the QP through the cvxpy lib and solve the problem for each subclass
        sk_means_2d_list = np.zeros(( chan_n, time_l, self.cluster_n))
        value = self.value
        for k in range(self.cluster_n):
            register_mean_2d = copy.deepcopy(data_2d_list)
            
            for t in range(self.cluster_n):
                if t == k:
                    register_mean_2d[..., t] = 0 * register_mean_2d[..., t]
                register_mean_2d[..., t] = value[k, t] * register_mean_2d[..., t]
            sk_means_2d_list[..., k] = (1 - np.sum(value[k,:])) * data_2d_list[..., k] + register_mean_2d.sum(axis = 2)
                
        return sk_means_2d_list, data_2d_list, value

class SKDCPM(JSSDCPM):
    
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
        Sb = sigma / time_len
        # calcute intra-class divergence matrix
        cov_all2 = np.zeros([chan_num, chan_num, trial_num])
        cov_all3 = np.zeros([chan_num, chan_num, trial_num])
        for n in range(target.shape[2]):
            cov_all2[..., n] = (target[..., n].squeeze() - template_tar) \
                                @ (target[..., n].squeeze() - template_tar).T
        cov_0 = np.mean(cov_all2, axis = 2)/time_len
        for n in range(nontarget.shape[2]):
            cov_all3[..., n] = (nontarget[..., n].squeeze() - template_nontar) \
                                @ (nontarget[..., n].squeeze() - template_nontar).T
        cov_1 = np.mean(cov_all3, axis = 2)
        sigma2 = (cov_0 + cov_1)/(2*time_len)
        Sw_pre = sigma2 / time_len
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        P = Sw_pre.shape[1]
        F = np.trace(Sw_pre)/P
        Tar = F * (np.eye(Sw_pre.shape[0]))
        shrink = shrinkage_method(trainset, Sw_pre, Tar)
        alpha, _ = shrink.oracle()
        Sw = (1 - alpha) * Sw_pre + alpha * F * (np.eye(Sw_pre.shape[0]))
        
        # solve the optimizatino problem
        svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
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

class MTS_DCPM(JSSDCPM):
    """Fit the MTS_DCPM model by fit_class function 
    """
    def __init__(        
        self, 
        component_num: int = 2,
        cluster_n = 3,
        shrinkage = True,
        p = 2, 
        shrinkage_type = None, 
        cov_norm = "l2",
        tol=10**(-5)* 5, 
        max_iter = 20
        ):

        self.cluster_n = cluster_n
        self.component_num = component_num
        self.shrinkage = shrinkage
        self.p = p
        self.shrinkage = shrinkage_type
        self.cov_norm = cov_norm
        self.cmp_num = component_num
        self.tol = tol
        self.max_iter = max_iter

    def _discriminant_eigenvalue(self, single_trial, S1m, S0m):
        """calculate the discriminate eigenvalue between
            filtered single trial data and the middle of interval
            model

        Args:
            single_trial (ndarray): shape[n_cmp, time_len]
            cmp_num (ndarray): component number used for test

        Returns:
            feature (ndarray): shape[n_cmp, ]  the discriminate 
                                eigenvalue of the inpute trial
        """
        R1 = (single_trial - S1m) @ (single_trial - S1m).T
        eigenvalue1 , _ = np.linalg.eig(R1)
        idx = np.argsort(eigenvalue1)
        eigenvalue1 =eigenvalue1[idx]
        
        R0 = (single_trial - S0m) @ (single_trial - S0m).T
        eigenvalue0 , _ = np.linalg.eig(R0)
        idx = np.argsort(eigenvalue0)
        eigenvalue0 =eigenvalue0[idx]
        
        feature = np.real(eigenvalue1)/(np.real(eigenvalue0)+10**(-25))
        return feature

    def _corr2(self, data, tmp):
        """calcute the 2d correlation coefficient same as matlab

        Args:
            data (numpy.ndarray): channel*time
            tmp (numpy.ndarray): channel*time

        Returns:
            corr2: 2d correlation coefficient
        """
        centra_data = data - data.mean()
        centra_tmp = tmp - tmp.mean()
        numerator = np.trace(centra_data @ centra_tmp.T)
        denominator = np.linalg.norm(centra_data, ord = 'fro')* np.linalg.norm(centra_tmp, ord = 'fro')
        corr2 = numerator/denominator
        return corr2
    
    def fit(self, trainset, dataset_label, dataset_labelsc, sub_class = 'intersect', whitening = False, ave_len = 1):
        """Fit the MTS_DCPM model by fit_class function 

        Args:
            trainset (numpy.ndarray): trial*channel*time
            dataset_label (numpy.ndarray): class label of each trial
            dataset_labelsc (numpy.ndarray): subclass label of each trial
            sub_class (str, optional):  if the number of subclass corresponding to each 
                                        class is different, then use the "intersect" pars. 
                                        Defaults to 'intersect'.
            whitening (bool, optional): weather whitening the dataset before shrinkage. 
                                        Defaults to False. Defaults to False.
            ave_len (int, optional): downsample coefficient. Defaults to 1.

        Returns:
            self
        """
            
        if sub_class == 'intersect':
            
            trial_n, chan_n, time_l = trainset.shape
            # get label type and intersect subclass label type
            target_label_type = np.unique(dataset_labelsc[dataset_label==1])
            ntarget_label_type = np.unique(dataset_labelsc[dataset_label==0])
            inter_label_type = np.intersect1d(target_label_type, ntarget_label_type)
            self.cluster_n = inter_label_type.size
            # initialize shrinkage estimator
            self.mts_tar = MTS_estimator()
            self.mts_ntar = MTS_estimator()
            # get the index in trail of inter subclass
            all_target_idx = np.argwhere(dataset_label == 1).squeeze()
            target_inter_idx = [i for i in all_target_idx if dataset_labelsc[i] in inter_label_type]
            all_ntarget_idx = np.argwhere(dataset_label == 0).squeeze()
            ntarget_inter_idx = [i for i in all_ntarget_idx if dataset_labelsc[i] in inter_label_type]
            # downsample 
            ds_trainset = ave_downsample(trainset, ave_len)
            # fit get shrinkage parameter
            self.mts_tar.fit(ds_trainset[target_inter_idx, ...], dataset_labelsc[target_inter_idx])
            self.tar_skmean2d, _, value1 = self.mts_tar.get_shrinkage_mean(trainset[target_inter_idx, ...])
            self.tar_skmean2d = np.concatenate((self.tar_skmean2d, trainset[dataset_label == 1,..., None].mean(axis = 0)), axis = 2)
            
            self.mts_ntar.fit(ds_trainset[ntarget_inter_idx, ...], dataset_labelsc[ntarget_inter_idx])
            self.ntar_skmean2d, _, value2 = self.mts_ntar.get_shrinkage_mean(trainset[ntarget_inter_idx, ...])
            self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, trainset[dataset_label == 0,..., None].mean(axis = 0)), axis = 2)
            
            self.filter_set = np.zeros((chan_n, chan_n, self.cluster_n + 1))
            label_type = inter_label_type
            
            for i, ltype in enumerate(label_type):
                ltype = int(ltype)
                select_idx = np.argwhere(dataset_labelsc == ltype).squeeze()
                subclass_data = trainset[select_idx, ...]
                trans_subclass_data = np.transpose(subclass_data, (1, 2, 0))
                subclass_label = trainset_label01[select_idx]
                self.filter_set[..., i] = self.fit_class(trans_subclass_data, subclass_label, self.tar_skmean2d[..., i], self.ntar_skmean2d[..., i])
                
            trans_subclass_data = np.transpose(trainset, (1, 2, 0))
            subclass_label = trainset_label01    
            self.filter_set[..., self.cluster_n] = self.fit_class(np.transpose(trainset, (1, 2, 0)), trainset_label01)

            for cmp_idx in range(chan_n):
                for i in range(self.cluster_n):
                    if self.filter_set[5:7,cmp_idx, i].sum() <=0:
                        self.filter_set[:,cmp_idx, i]  = self.filter_set[:,cmp_idx, i] * -1
                        
            # plt.plot(self.filter_set[:,1, :])
            # plt.show()
            k = 1
        else:
            
            self.mts = MTS_estimator()
            trial_n, chan_n, time_l = trainset.shape
            trainset_label = dataset_label.copy()
            label_type = np.unique(dataset_labelsc[trainset_label==1])
            self.cluster_n  = label_type.size
            self.mts.fit(trainset[trainset_label ==1, ...], dataset_labelsc[trainset_label ==1])
            self.ntar_2d = trainset[trainset_label==0,...].mean(axis = 0)
            self.tar_skmean2d, _, value = self.mts.get_shrinkage_mean(trainset[trainset_label ==1, ...])
            self.tar_skmean2d = np.concatenate((self.tar_skmean2d, trainset[trainset_label==1,..., None].mean(axis = 0)), axis = 2)
            self.ntar_skmean2d = self.ntar_2d
            self.filter_set = np.zeros((chan_n, chan_n, self.cluster_n + 1))
            
            for i, ltype in enumerate(label_type):
                ltype = int(ltype)
                select_idx_t = np.argwhere(dataset_labelsc == ltype).squeeze()
                select_idx_nt = np.argwhere(trainset_label == 0).squeeze()
                select_idx = np.concatenate((select_idx_t, select_idx_nt))
                subclass_data = trainset[select_idx, ...]
                trans_subclass_data = np.transpose(subclass_data, (1, 2, 0))
                subclass_label = trainset_label[select_idx]
                self.filter_set[..., i] = self.fit_class(trans_subclass_data, subclass_label, self.tar_skmean2d[..., i], self.ntar_2d)
                
            trans_subclass_data = np.transpose(trainset, (1, 2, 0))
            subclass_label = trainset_label    
            self.filter_set[..., self.cluster_n] = self.fit_class(np.transpose(trainset, (1, 2, 0)), trainset_label)


            for cmp_idx in range(chan_n):
                for i in range(self.cluster_n):
                    if self.filter_set[5:7,cmp_idx, i].sum() <=0:
                        self.filter_set[:,cmp_idx, i]  = self.filter_set[:,cmp_idx, i] * -1
        
        return self
     
    def fit_class1(self,subclass_data, subclass_label, target_mean = None, nontarget_mean = None):
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
        trainset = subclass_data
        trainset_label = subclass_label
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        
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
        if (target_mean is None) and (nontarget_mean is None):
            template_target =  target_trial.mean(axis = 2)       # extract target template
            template_nontarget = nontarget_trial.mean(axis = 2) # extract nontarget template
            # self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, template_nontar[..., None]), axis = 2)
            # self.tar_skmean2d = np.concatenate((self.tar_skmean2d, template_tar[..., None]), axis = 2)
            
        else:
            template_target =  target_mean       # extract target template
            template_nontarget = nontarget_mean # extract nontarget template
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
        for k in range(maximum_iteration):

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
        self.filter = np.real(W)

        return self.filter 

    def fit_class(self, subclass_data, subclass_label, target_mean = None, nontarget_mean = None):
        """fit the typical DSP spatial filter according to given sample mean

        Args:
            subclass_data (numpy.ndarray): trial*channel*time
            subclass_label (numpy.ndarray): trial
            target_mean (numpy.ndarray, optional): given sample mean of target class. Defaults to None.
            nontarget_mean (numpy.ndarray, optional):given sample mean of nontarget class. Defaults to None.

        Returns:
            self.filter: the spatial  filter
        """        
        trainset = subclass_data
        trainset_label = subclass_label
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        
        # data centralization
        for i in range(trainset.shape[2]):

            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
        trainset = train_trial
        # get class trial
        target = trainset[..., trainset_label.squeeze() == 1]
        nontarget = trainset[..., trainset_label.squeeze() == 0]
        # get class template
        if (target_mean is None) and (nontarget_mean is None):
            template_tar =  target.mean(axis = 2)       # extract target template
            template_nontar = nontarget.mean(axis = 2) # extract nontarget template
        else:
            template_tar =  target_mean       # extract target template
            template_nontar = nontarget_mean # extract nontarget template
            
        template_all = (template_tar + template_nontar) / 2
        # calcute  between-class divergence matrix
        sigma = ((template_tar - template_all) @ (template_tar - template_all).T \
                + (template_nontar - template_all) @ (template_nontar - template_all).T)/2 
        Sb = sigma/time_len
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
        Sw_pre = (cov_0 + cov_1)/(2*time_len)
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        
        if self.shrinkage is False:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            alpha = 0
        else:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            alpha, _ = shrink.oracle()
            alpha = alpha
       
        Sw = (1 - alpha) * Sw_pre + alpha * Tar
        
        # solve the optimizatino problem
        svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W = right_vector[:,denote_idx]
        # save DCPM model
        self.filter = np.real(sorted_W)
        self.target_tmp = template_tar
        self.nontarget_tmp = template_nontar
        return self.filter
  
    def transform(self, dataset, filter, cmp_num):
        """transform origin data to filtered data

        Args:
            dataset (ndarry): chan_num*time_len*trial_num
            cmp_num (int): the number of component used for filter

        Returns:
            ndarry: filtered datatset
        """
        _, time_len, trial_num = dataset.shape
        filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
        W = filter[:,0:cmp_num]
        for i in range(trial_num):
            filtered_dataset[..., i] = W.T @ dataset[..., i] 
        return filtered_dataset
    
    def feature_extract(self, testset_o, cmp_num, ensemble = False):
        """extract feature of dataset, here we provide two kinds of
        method.
            The first method, called ensemble method, is the same as the 
        feature extract procedure in TRCA witch concatenate the filter belonging 
        to different subclass to filter the data. Using the difference between 
        the Euclidean distance from the sample to the positive template and the 
        sample to the negative template as the extracted feature.
            The second method is the same as the feature extract procedure in DACIE.
        Args:
            testset_o (numpy.ndarray): channel*time*trial
            cmp_num (int): the filter dimension used for feature extracting
            ensamble (bool, optional): weather using ensembel. Defaults to False.

        Returns:
            predict_label: trial
            criterion: criterion using to get predict_label
        """        
        testset = np.transpose(testset_o, (1, 2, 0))
        # centralization
        location = np.mean(testset, axis = 1, keepdims = True)
        testset = testset - location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.tar_skmean2d - self.tar_skmean2d.mean(axis = 1, keepdims=True)
        template_nontar =  self.ntar_skmean2d - self.ntar_skmean2d.mean(axis = 1, keepdims=True)
        # get filtered class template
        
        self.template_1 = list()
        self.template_0 = list()
        ftestset = list()
        DSP_filter = self.filter_set
        tmp_num =  self.cluster_n +1
        self.criterion = np.zeros((trial_num, tmp_num))
        self.predict_label = np.zeros((trial_num, tmp_num))
        
        self.feature = np.zeros((trial_num, (tmp_num)*cmp_num))
        self.feature1 = np.zeros((trial_num, cmp_num, tmp_num))
        
        if ensemble is True:
            
            self.ensamble_criterion = np.zeros((trial_num,))        
            self.ensamble_predict_label = np.zeros((trial_num,))
            # ensambel method
            ensamble_filter = DSP_filter[:, :cmp_num, 0]
            cat_template_tar = template_tar[..., 0]
            cat_template_nontar = template_nontar[..., 0]
            cat_testset = testset
            for i in range(tmp_num-1):
                ensamble_filter = np.concatenate((ensamble_filter, DSP_filter[:, :cmp_num, i+1]), axis = 1)
                cat_template_tar = np.concatenate((cat_template_tar, template_tar[..., i+1]), axis = 1)
                cat_template_nontar = np.concatenate((cat_template_nontar, template_nontar[..., i+1]), axis = 1)
                cat_testset = np.concatenate((cat_testset, testset), axis = 1)
                
            self.cat_template_1 = ensamble_filter.T @ cat_template_tar
            self.cat_template_0 = ensamble_filter.T @ cat_template_nontar
            ensambel_ftestset = self.transform(cat_testset, ensamble_filter, cmp_num*tmp_num)
        
            
            for j in range(trial_num):
                
                filtered_trial = ensambel_ftestset[..., j] 
                dist_ntar = np.linalg.norm(self.cat_template_0  - filtered_trial)**2
                dist_tar = np.linalg.norm(self.cat_template_1 - filtered_trial)**2
                    
                self.ensamble_criterion[j] = dist_ntar - dist_tar

            self.ensamble_predict_label = (np.sign(self.ensamble_criterion) + 1) / 2

            return self.ensamble_predict_label, self.ensamble_criterion
            
        else:
            
            for i in range(tmp_num):

                self.template_1.append(DSP_filter[:, :cmp_num, i].T @ template_tar[..., i])
                self.template_0.append(DSP_filter[:, :cmp_num, i].T @ template_nontar[...,i])
                # plt.plot(self.template_1[0].T)
                # plt.show()
                
                # get filtered data
                ftestset.append(self.transform(testset, DSP_filter[:, :cmp_num, i], cmp_num))
                
                # classification
                for j in range(trial_num):
                    
                    filtered_trial = ftestset[i][..., j] 
                    dist_ntar = np.linalg.norm(self.template_0[i] - filtered_trial)**2
                    dist_tar = np.linalg.norm(self.template_1[i] - filtered_trial)**2
                    self.criterion[j, i] = dist_ntar - dist_tar
                    self.feature[j, i*cmp_num : (i+1)*cmp_num] = self._discriminant_eigenvalue(filtered_trial, self.template_1[i], self.template_0[i])
                    # self.feature1[j, :, i] = self._discriminant_eigenvalue(filtered_trial, self.template_1[i], self.template_0[i])
                
                # statistic classification accuracy
                self.predict_label[:, i] = (np.sign(self.criterion[:, i]) + 1) / 2
            
            return self.criterion, self.predict_label
            
    def predict(self, trainset, trainset_label, testset, cmp_num, ensamble = False):
        """predict the trial in testset

        Args:
            trainset (numpy.ndarray): channel*time*trial
            trainset_label (numpy.ndarray): trial
            testset (numpy.ndarray): channel*time*trial
            cmp_num (numpy.ndarray): filter dimension using for feature extraction
            ensamble (bool, optional): weather using ensembel. Defaults to False.
            
        Returns:
            self.predict_label: _description_
            self.criterion: criterion using to get predict_label
        """        
        if ensamble is True:
            self.ensamble_label, self.ensamble_criterion = self.feature_extract(testset, cmp_num, ensamble = ensamble)
            return self.ensamble_label, self.ensamble_criterion
        else:
            trainset_label[trainset_label!=0] = 1
            train_feature, _= self.feature_extract(trainset, cmp_num)
            test_feature, plabel = self.feature_extract(testset, cmp_num)
            # preprocessing
            scaler =  MinMaxScaler()
            ztrain_eig_feature = scaler.fit_transform(train_feature[:,:-1])
            ztest_eig_feature  = scaler.transform(test_feature[:,:-1])

            # classification
            self.clf = SVC(
                C= 2.0, 
                kernel="linear", 
                degree= 3 , 
                gamma='scale', 
                coef0= 0.0001, 
                shrinking=True, 
                probability=False, 
                tol=0.001, 
                cache_size=1000, 
                class_weight= 'balanced', 
                verbose=False, 
                max_iter=1000, 
                decision_function_shape='ovo', 
                break_ties=False, 
                random_state=1079)

            self.clf.fit(ztrain_eig_feature, trainset_label)
            self.criterion = self.clf.decision_function(ztest_eig_feature)
            self.predict_label = self.clf.predict(ztest_eig_feature)
            # return self.predict_label, self.criterion
            return plabel, test_feature
        
    def score(self, trainset, trainset_label, testset, testset_label, cmp_num, TPlabel = 1, TNlabel = 0, ensamble = False):
        testset_label[testset_label != 0] = 1

        predict_label, criterion = self.predict(trainset, trainset_label, testset, cmp_num, ensamble = ensamble)
        result_1 = list()
        if ensamble is False:
            for i in range(self.cluster_n + 1):
                result_1.append(super().cal_score(testset_label, predict_label[:,i], criterion[:,i], TPlabel, TNlabel) )
            return result_1
        else:
            return super().cal_score(testset_label, predict_label, criterion, TPlabel, TNlabel)

class MTS_DCPM_c(JSSDCPM):
    """Fit the MTS_DCPM model, the filter was calucated by fit_class_all_subclass function 
    """
    def __init__(        
        self, 
        component_num: int = 2,
        cluster_n = 3,
        shrinkage = True,
        p = 2, 
        shrinkage_type = None, 
        cov_norm = "l2",
        tol=10**(-5)* 5, 
        max_iter = 20
        ):

        self.cluster_n = cluster_n
        self.component_num = component_num
        self.shrinkage = shrinkage
        self.p = p
        self.shrinkage = shrinkage_type
        self.cov_norm = cov_norm
        self.cmp_num = component_num
        self.tol = tol
        self.max_iter = max_iter

    def _discriminant_eigenvalue(self, single_trial, S1m, S0m):
        """calculate the discriminate eigenvalue between
            filtered single trial data and the middle of interval
            model

        Args:
            single_trial (ndarray): shape[n_cmp, time_len]
            cmp_num (ndarray): component number used for test

        Returns:
            feature (ndarray): shape[n_cmp, ]  the discriminate 
                                eigenvalue of the inpute trial
        """
        R1 = (single_trial - S1m) @ (single_trial - S1m).T
        eigenvalue1 , _ = np.linalg.eig(R1)
        idx = np.argsort(eigenvalue1)
        eigenvalue1 =eigenvalue1[idx]
        
        R0 = (single_trial - S0m) @ (single_trial - S0m).T
        eigenvalue0 , _ = np.linalg.eig(R0)
        idx = np.argsort(eigenvalue0)
        eigenvalue0 =eigenvalue0[idx]
        
        feature = np.real(eigenvalue1)/(np.real(eigenvalue0)+10**(-25))
        return feature

    def _corr2(self, data, tmp):
        """calcute the 2d correlation coefficient same as matlab

        Args:
            data (numpy.ndarray): channel*time
            tmp (numpy.ndarray): channel*time

        Returns:
            corr2: 2d correlation coefficient
        """        
        centra_data = data - data.mean()
        centra_tmp = tmp - tmp.mean()
        numerator = np.trace(centra_data @ centra_tmp.T)
        denominator = np.linalg.norm(centra_data, ord = 'fro')* np.linalg.norm(centra_tmp, ord = 'fro')
        corr2 = numerator/denominator
        return corr2
    
    def fit(self, trainset, dataset_label, dataset_labelsc, sub_class = 'intersect', whitening = False, ave_len = 1):
        """Fit the MTS_DCPM model, the filter was calucated by fit_class_all_subclass function 

        Args:
            trainset (numpy.ndarray): trial*channel*time
            dataset_label (numpy.ndarray): class label of each trial
            dataset_labelsc (numpy.ndarray): subclass label of each trial
            sub_class (str, optional):  if the number of subclass corresponding to each 
                                        class is different, then use the "intersect" pars. 
                                        Defaults to 'intersect'.
            whitening (bool, optional): weather whitening the dataset before shrinkage. 
                                        Defaults to False. Defaults to False.
            ave_len (int, optional): downsample coefficient. Defaults to 1.

        Returns:
            self
        """
        if sub_class == 'intersect':
            
            trial_n, chan_n, time_l = trainset.shape
            # get label type and intersect subclass label type
            target_label_type = np.unique(dataset_labelsc[dataset_label==1])
            ntarget_label_type = np.unique(dataset_labelsc[dataset_label==0])
            inter_label_type = np.intersect1d(target_label_type, ntarget_label_type)
            self.cluster_n = inter_label_type.size
            # initialize shrinkage estimator
            self.mts_tar = MTS_estimator()
            self.mts_ntar = MTS_estimator()
            # get the index in trail of inter subclass
            all_target_idx = np.argwhere(dataset_label == 1).squeeze()
            target_inter_idx = [i for i in all_target_idx if dataset_labelsc[i] in inter_label_type]
            all_ntarget_idx = np.argwhere(dataset_label == 0).squeeze()
            ntarget_inter_idx = [i for i in all_ntarget_idx if dataset_labelsc[i] in inter_label_type]
            # downsample 
            ds_trainset = ave_downsample(trainset, ave_len)
            # fit get shrinkage parameter
            self.mts_tar.fit(ds_trainset[target_inter_idx, ...], dataset_labelsc[target_inter_idx], whitening=  whitening)
            self.tar_skmean2d, _, value1 = self.mts_tar.get_shrinkage_mean(trainset[target_inter_idx, ...])
            self.tar_skmean2d = np.concatenate((self.tar_skmean2d, trainset[dataset_label == 1,..., None].mean(axis = 0)), axis = 2)
            
            self.mts_ntar.fit(ds_trainset[ntarget_inter_idx, ...], dataset_labelsc[ntarget_inter_idx], whitening=  whitening)
            self.ntar_skmean2d, _, value2 = self.mts_ntar.get_shrinkage_mean(trainset[ntarget_inter_idx, ...])
            self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, trainset[dataset_label == 0,..., None].mean(axis = 0)), axis = 2)
            
            self.filter_set = np.zeros((chan_n, chan_n, self.cluster_n + 1))
            label_type = inter_label_type
            
            for i, ltype in enumerate(label_type):
                ltype = int(ltype)
                select_idx = np.argwhere(dataset_labelsc == ltype).squeeze()
                self.filter_set[..., i] = self.fit_class_all_subclass(trainset, dataset_label, select_idx, self.tar_skmean2d[..., i], self.ntar_skmean2d[..., i])                
            self.filter_set[..., self.cluster_n] = self.fit_class(np.transpose(trainset, (1, 2, 0)), trainset_label01)

            for cmp_idx in range(chan_n):
                for i in range(self.cluster_n):
                    if self.filter_set[5:7,cmp_idx, i].sum() <=0:
                        self.filter_set[:,cmp_idx, i]  = self.filter_set[:,cmp_idx, i] * -1
                        
        else:
            
            self.mts = MTS_estimator()
            trial_n, chan_n, time_l = trainset.shape
            trainset_label = dataset_label.copy()
            label_type = np.unique(dataset_labelsc[trainset_label==1])
            self.cluster_n  = label_type.size
            self.mts.fit(trainset[trainset_label ==1, ...], dataset_labelsc[trainset_label ==1])
            self.ntar_2d = trainset[trainset_label==0,...].mean(axis = 0)
            self.tar_skmean2d, _, value = self.mts.get_shrinkage_mean(trainset[trainset_label ==1, ...])
            self.tar_skmean2d = np.concatenate((self.tar_skmean2d, trainset[trainset_label==1,..., None].mean(axis = 0)), axis = 2)
            self.ntar_skmean2d = self.ntar_2d
            self.filter_set = np.zeros((chan_n, chan_n, self.cluster_n + 1))
            
            for i, ltype in enumerate(label_type):
                ltype = int(ltype)
                select_idx_t = np.argwhere(dataset_labelsc == ltype).squeeze()
                select_idx_nt = np.argwhere(trainset_label == 0).squeeze()
                select_idx = np.concatenate((select_idx_t, select_idx_nt))
                self.filter_set[..., i] = self.fit_class_all_subclass(trainset, dataset_label, select_idx, self.tar_skmean2d[..., i], self.ntar_skmean2d[..., i])         
            self.filter_set[..., self.cluster_n] = self.fit_class(np.transpose(trainset, (1, 2, 0)), trainset_label)
            
            for cmp_idx in range(chan_n):
                for i in range(self.cluster_n):
                    if self.filter_set[5:7,cmp_idx, i].sum() <=0:
                        self.filter_set[:,cmp_idx, i]  = self.filter_set[:,cmp_idx, i] * -1
        
        return self
  
    def fit_class(self, subclass_data, subclass_label, target_mean = None, nontarget_mean = None):
        """fit the SKDSP spatial filter according to given sample mean

        Args:
            subclass_data (numpy.ndarray): trial*channel*time
            subclass_label (numpy.ndarray): trial
            target_mean (numpy.ndarray, optional): given sample mean of target class. Defaults to None.
            nontarget_mean (numpy.ndarray, optional):given sample mean of nontarget class. Defaults to None.

        Returns:
            self.filter: the spatial  filter
        """
        trainset = subclass_data
        trainset_label = subclass_label
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        
        # data centralization
        for i in range(trainset.shape[2]):

            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
        trainset = train_trial
        # get class trial
        target = trainset[..., trainset_label.squeeze() == 1]
        nontarget = trainset[..., trainset_label.squeeze() == 0]
        # get class template
        if (target_mean is None) and (nontarget_mean is None):
            template_tar =  target.mean(axis = 2)       # extract target template
            template_nontar = nontarget.mean(axis = 2) # extract nontarget template

        else:
            template_tar =  target_mean       # extract target template
            template_nontar = nontarget_mean # extract nontarget template
            
        template_all = (template_tar + template_nontar) / 2
        # calcute  between-class divergence matrix
        sigma = ((template_tar - template_all) @ (template_tar - template_all).T \
                + (template_nontar - template_all) @ (template_nontar - template_all).T)/2 
        Sb = sigma/time_len
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
        Sw_pre = (cov_0 + cov_1)/(2*time_len)
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        
        if self.shrinkage is False:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            alpha = 0
        else:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            alpha, _ = shrink.oracle()
            alpha = alpha
       
        Sw = (1 - alpha) * Sw_pre + alpha * Tar
        
        # solve the optimizatino problem
        svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W = right_vector[:,denote_idx]
        # save DCPM model
        self.filter = np.real(sorted_W)
        self.target_tmp = template_tar
        self.nontarget_tmp = template_nontar
        return self.filter

    def fit_class_all_subclass(self, trainset, dataset_label, target_mean = None, nontarget_mean = None):
        """fit the SKDSP spatial filter according to given sample mean meanwhile
            using all data in trainset to calculate within-class covariance
        Args:
            trainset (numpy.ndarray): trial*channel*time
            dataset_label (numpy.ndarray): trial
            target_mean (numpy.ndarray):  given sample mean of target class. Defaults to None.
            nontarget_mean (numpy.ndarray):  given sample mean of nontarget class. Defaults to None.

        Returns:
            self.filter: the spatial  filter
        """
        trainset = np.transpose(trainset, (1, 2, 0))
        trainset_label = dataset_label
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        
        # data centralization
        for i in range(trainset.shape[2]):

            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
            
        # get class trial
        target = trainset[..., trainset_label.squeeze() == 1]
        nontarget = trainset[..., trainset_label.squeeze() == 0]
        # get class template
        if (target_mean is None) and (nontarget_mean is None):
            template_tar =  target.mean(axis = 2)       # extract target template
            template_nontar = nontarget.mean(axis = 2) # extract nontarget template
            # self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, template_nontar[..., None]), axis = 2)
            # self.tar_skmean2d = np.concatenate((self.tar_skmean2d, template_tar[..., None]), axis = 2)
            
        else:
            template_tar =  target_mean       # extract target template
            template_nontar = nontarget_mean # extract nontarget template
            
        template_all = (template_tar + template_nontar) / 2
        # calcute  between-class divergence matrix
        sigma = ((template_tar - template_all) @ (template_tar - template_all).T \
                + (template_nontar - template_all) @ (template_nontar - template_all).T)/2 
        Sb = sigma/time_len
        # calcute intra-class divergence matrix
        label_type = np.unique(dataset_label)
        
        cov_set = list()
        for type_idx, type_name in enumerate(label_type):
            idx = np.argwhere(trainset_label==type_name)
            cov_all = 0
            if type_name == 0:
                for n in range(idx.size):
                    cov_all += (trainset[..., idx[n]].squeeze() - template_nontar) \
                                @ (trainset[..., idx[n]].squeeze() - template_nontar).T
                cov_all = cov_all/(idx.size * time_len)
                cov_set.append(cov_all)  
            elif type_name !=0:
                for n in range(idx.size):
                    cov_all += (trainset[..., idx[n]].squeeze() - template_tar) \
                                        @ (trainset[..., idx[n]].squeeze() - template_tar).T    
                cov_all = cov_all/(idx.size * time_len)
                cov_set.append(cov_all)  
        cov_array = np.array(cov_set)
        Sw_pre = np.mean(cov_array, axis = 0)
        ## calcute the shrinkage coefficient and shrinkage Sw
        
        if self.shrinkage is False:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            alpha = 0
        else:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            alpha, _ = shrink.oracle()
            alpha = alpha
       
        Sw = (1 - alpha) * Sw_pre + alpha * Tar
        
        # solve the optimizatino problem
        svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W = right_vector[:,denote_idx]
        # save DCPM model
        self.filter = np.real(sorted_W)
        self.target_tmp = template_tar
        self.nontarget_tmp = template_nontar
        return self.filter
  
    def transform(self, dataset, filter, cmp_num):
        """transform origin data to filtered data

        Args:
            dataset (ndarry): chan_num*time_len*trial_num
            cmp_num (int): the number of component used for filter

        Returns:
            ndarry: filtered datatset
        """
        _, time_len, trial_num = dataset.shape
        filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
        W = filter[:,0:cmp_num]
        for i in range(trial_num):
            filtered_dataset[..., i] = W.T @ dataset[..., i] 
        return filtered_dataset
    
    def feature_extract(self, testset_o, cmp_num, ensemble = False):
        """extract feature of dataset, here we provide two kinds of
        method.
            The first method, called ensemble method, is the same as the 
        feature extract procedure in TRCA witch concatenate the filter belonging 
        to different subclass to filter the data. Using the difference between 
        the Euclidean distance from the sample to the positive template and the 
        sample to the negative template as the extracted feature.
            The second method is the same as the feature extract procedure in DACIE.
        Args:
            testset_o (numpy.ndarray): channel*time*trial
            cmp_num (int): the filter dimension used for feature extracting
            ensamble (bool, optional): weather using ensembel. Defaults to False.

        Returns:
            predict_label: trial
            criterion: criterion using to get predict_label
        """
        testset = np.transpose(testset_o, (1, 2, 0))
        # centralization
        location = np.mean(testset, axis = 1, keepdims = True)
        testset = testset - location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.tar_skmean2d - self.tar_skmean2d.mean(axis = 1, keepdims=True)
        template_nontar =  self.ntar_skmean2d - self.ntar_skmean2d.mean(axis = 1, keepdims=True)
        # get filtered class template
        
        self.template_1 = list()
        self.template_0 = list()
        ftestset = list()
        DSP_filter = self.filter_set
        tmp_num =  self.cluster_n +1
        self.criterion = np.zeros((trial_num, tmp_num))
        self.predict_label = np.zeros((trial_num, tmp_num))
        
        self.feature = np.zeros((trial_num, (tmp_num)*cmp_num))
        self.feature1 = np.zeros((trial_num, cmp_num, tmp_num))
        
        if ensemble is True:
            
            self.ensemble_criterion = np.zeros((trial_num,))        
            self.ensemble_predict_label = np.zeros((trial_num,))
            # ensambel method
            ensemble_filter = DSP_filter[:, :cmp_num, 0]
            cat_template_tar = template_tar[..., 0]
            cat_template_nontar = template_nontar[..., 0]
            cat_testset = testset
            for i in range(tmp_num-1):
                ensemble_filter = np.concatenate((ensemble_filter, DSP_filter[:, :cmp_num, i+1]), axis = 1)
                cat_template_tar = np.concatenate((cat_template_tar, template_tar[..., i+1]), axis = 1)
                cat_template_nontar = np.concatenate((cat_template_nontar, template_nontar[..., i+1]), axis = 1)
                cat_testset = np.concatenate((cat_testset, testset), axis = 1)
   
            self.cat_template_1 = ensemble_filter.T @ cat_template_tar
            self.cat_template_0 = ensemble_filter.T @ cat_template_nontar
            ensambel_ftestset = self.transform(cat_testset, ensemble_filter, cmp_num*tmp_num)
        
            
            for j in range(trial_num):
                
                filtered_trial = ensambel_ftestset[..., j] 
                dist_ntar = np.linalg.norm(self.cat_template_0  - filtered_trial)**2
                dist_tar = np.linalg.norm(self.cat_template_1 - filtered_trial)**2
                    
                self.ensemble_criterion[j] = dist_ntar - dist_tar

            self.ensemble_predict_label = (np.sign(self.ensemble_criterion) + 1) / 2

            return self.ensemble_predict_label, self.ensemble_criterion
            
        else:
            
            for i in range(tmp_num):

                self.template_1.append(DSP_filter[:, :cmp_num, i].T @ template_tar[..., i])
                self.template_0.append(DSP_filter[:, :cmp_num, i].T @ template_nontar[...,i])
                
                # get filtered data
                ftestset.append(self.transform(testset, DSP_filter[:, :cmp_num, i], cmp_num))
                
                # classification
                for j in range(trial_num):
                    
                    filtered_trial = ftestset[i][..., j] 
                    dist_ntar = np.linalg.norm(self.template_0[i] - filtered_trial)**2
                    dist_tar = np.linalg.norm(self.template_1[i] - filtered_trial)**2
                    self.criterion[j, i] = dist_ntar - dist_tar
                    self.feature[j, i*cmp_num : (i+1)*cmp_num] = self._discriminant_eigenvalue(filtered_trial, self.template_1[i], self.template_0[i])
                    # self.feature1[j, :, i] = self._discriminant_eigenvalue(filtered_trial, self.template_1[i], self.template_0[i])
                
                # statistic classification accuracy
                self.predict_label[:, i] = (np.sign(self.criterion[:, i]) + 1) / 2
            self.predict_label = self.predict_label.sum(axis = 1, keepdims = True)
            self.predict_label[self.predict_label>=3] = 1
            return self.criterion, self.predict_label
            
    def predict(self, trainset, trainset_label, testset, cmp_num, ensamble = False):
        """predict the trial in testset

        Args:
            trainset (numpy.ndarray): channel*time*trial
            trainset_label (numpy.ndarray): trial
            testset (numpy.ndarray): channel*time*trial
            cmp_num (numpy.ndarray): filter dimension using for feature extraction
            ensamble (bool, optional): weather using ensembel. Defaults to False.
            
        Returns:
            self.predict_label: _description_
            self.criterion: criterion using to get predict_label
        """
        if ensamble is True:
            self.ensamble_label, self.ensamble_criterion = self.feature_extract(testset, cmp_num, ensamble = ensamble)
            return self.ensamble_label, self.ensamble_criterion
        else:
            trainset_label[trainset_label!=0] = 1
            train_feature, _= self.feature_extract(trainset, cmp_num)
            test_feature, plabel = self.feature_extract(testset, cmp_num)
            # preprocessing
            scaler =  MinMaxScaler()
            ztrain_eig_feature = scaler.fit_transform(train_feature[:,:-1])
            ztest_eig_feature  = scaler.transform(test_feature[:,:-1])

            # classification
            self.clf = SVC(
                C= 2.0, 
                kernel="linear", 
                degree= 3 , 
                gamma='scale', 
                coef0= 0.0001, 
                shrinking=True, 
                probability=False, 
                tol=0.001, 
                cache_size=1000, 
                class_weight= 'balanced', 
                verbose=False, 
                max_iter=1000, 
                decision_function_shape='ovo', 
                break_ties=False, 
                random_state=1079)

            self.clf.fit(ztrain_eig_feature, trainset_label)
            self.criterion = self.clf.decision_function(ztest_eig_feature)
            self.predict_label = self.clf.predict(ztest_eig_feature)
            return self.predict_label, self.criterion
        
    def score(self, trainset, trainset_label, testset, testset_label, cmp_num, TPlabel = 1, TNlabel = 0, ensamble = False):
        testset_label[testset_label != 0] = 1

        predict_label, criterion = self.predict(trainset, trainset_label, testset, cmp_num, ensamble = ensamble)
        result_1 = list()
        if ensamble is False:
            for i in range(predict_label.shape[1]):
                result_1.append(super().cal_score(testset_label, predict_label[:,i], criterion[:,i], TPlabel, TNlabel) )
            return result_1
        else:
            return super().cal_score(testset_label, predict_label, criterion, TPlabel, TNlabel)

class srMTL():
    
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y):
        """fit the subclass relationship multi-task learning model

        Args:
            X (numpy.ndarray): trial * feature
            y (numpy.ndarray): trial

        Returns:
            self: self.select_idx is the selected feature elements
            
        Zhang, Y., Zhou, T., Wu, W., Xie, H., Zhu, H., Zhou, G., & Cichocki, A. (2022). 
        Improving EEG Decoding via Clustering-Based Multitask Feature Learning. 
        IEEE Transactions on Neural Networks and Learning Systems, 33(8), 3587–3597. 
        https://doi.org/10.1109/TNNLS.2021.3053576
        """
        N, Nnc = y.shape
        N_feature = X.shape[1]
        # 计算类别间相似度矩阵
        S = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if np.sum(np.abs(y[i, :] - y[j, :])) == 0:#np.array_equal(A,B) 
                    S[i, j] = 1
                else:
                    S[i, j] = 0
        D = np.diag(S.sum(axis = 1))
        L = D - S
        # 初始化基本参数
        p = 1
        m = Nnc
        lambda1 = 0.01
        lambda2 = 0.5
        max_iteration = 200
        stop_criterion = 5*10**(-15)
        self.W = np.diag(np.ones((N_feature)))[:,0:m]
        W_t_1 = self.W
        # 初始化L21范数转换矩阵
        ttt = 10
        xi = 10**(-ttt)
        xx = np.linalg.norm(self.W, ord = 2, axis = 1)** (2-p)
        xx[np.isinf(xx)] = 10**20
        E = np.diag(1/(xx + xi) ) 
        
        # iteration 
        wt_register = list()
        criterion_register = list()
        criterion_register.append(100)
        for i in range(max_iteration):
            
            self.W = np.linalg.inv(X.T  @ X + lambda1*E + lambda2*X.T @ L @ X) @ X.T @ y

            xx = np.linalg.norm(self.W, ord = 2, axis = 1)** (2-p)
            xx[np.isinf(xx)] = 10**80
            E = np.diag(1/(xx + xi) ) 

            wt_register.append(self.W)
            loss =  np.linalg.norm(self.W - W_t_1)#/np.max([np.linalg.norm(self.W), np.linalg.norm(W_t_1)])
            criterion_register.append(loss)
            loss_var = np.abs(criterion_register[-2] - criterion_register[-1])/np.abs(criterion_register[-2])
            # print(loss)
            # print(loss_var)
            #refresh data
            W_t_1= self.W
            if (loss < stop_criterion) or (loss_var <0.05):
                break
            
        self.W_abs = np.abs(self.W)   
        mean_W = self.W.mean()     
        kkk =self.W<mean_W/100
        self.W_abs[kkk.sum(axis = 1)>=Nnc/1.5, :] = np.zeros((m,))
        self.W_abs[kkk.sum(axis = 1)<Nnc/1.5, :] = np.ones((m,))
        self.select_idx = np.argwhere(self.W_abs[:,0]==0)
        return self

class MTS_MTL_DCPM_c(JSSDCPM):

    def __init__(        
        self, 
        component_num: int = 2,
        cluster_n = 3,
        shrinkage = True,
        p = 2, 
        shrinkage_type = None, 
        cov_norm = "l2",
        tol=10**(-5)* 5, 
        max_iter = 20
        ):
        self.cluster_n = cluster_n
        self.component_num = component_num
        self.shrinkage = shrinkage
        self.p = p
        self.shrinkage = shrinkage_type
        self.cov_norm = cov_norm
        self.cmp_num = component_num
        self.tol = tol
        self.max_iter = max_iter

    def _discriminant_eigenvalue(self, single_trial, S1m, S0m):
        """calculate the discriminate eigenvalue between
            filtered single trial data and the middle of interval
            model

        Args:
            single_trial (ndarray): shape[n_cmp, time_len]
            cmp_num (ndarray): component number used for test

        Returns:
            feature (ndarray): shape[n_cmp, ]  the discriminate 
                                eigenvalue of the inpute trial
        """
        R1 = (single_trial - S1m) @ (single_trial - S1m).T
        eigenvalue1 , _ = np.linalg.eig(R1)
        idx = np.argsort(eigenvalue1)
        eigenvalue1 =eigenvalue1[idx]
        
        R0 = (single_trial - S0m) @ (single_trial - S0m).T
        eigenvalue0 , _ = np.linalg.eig(R0)
        idx = np.argsort(eigenvalue0)
        eigenvalue0 =eigenvalue0[idx]
        
        feature = np.real(eigenvalue1)/(np.real(eigenvalue0)+10**(-25))
        return feature

    def _corr2(self, data, tmp):
        """calcute the 2d correlation coefficient same as matlab

        Args:
            data (numpy.ndarray): channel*time
            tmp (numpy.ndarray): channel*time

        Returns:
            corr2: 2d correlation coefficient
        """        
        centra_data = data - data.mean()
        centra_tmp = tmp - tmp.mean()
        numerator = np.trace(centra_data @ centra_tmp.T)
        denominator = np.linalg.norm(centra_data, ord = 'fro')* np.linalg.norm(centra_tmp, ord = 'fro')
        corr2 = numerator/denominator
        return corr2

    def _fit_MTS_DSP_intersect(self, trainset, dataset_label, dataset_labelsc, whitening = False, ave_len = 1):
        """_summary_

        Args:
            trainset (_type_): _description_
            dataset_label (_type_): _description_
            dataset_labelsc (_type_): _description_
            whitening (bool, optional): _description_. Defaults to False.
            ave_len (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        trial_n, chan_n, time_l = trainset.shape
        # get label type and intersect subclass label type
        target_label_type = np.unique(dataset_labelsc[dataset_label==1])
        ntarget_label_type = np.unique(dataset_labelsc[dataset_label==0])
        inter_label_type = np.intersect1d(target_label_type, ntarget_label_type)
        self.inter_label_type = inter_label_type 
        self.cluster_n = inter_label_type.size
        self.inter_idx = list()
        for i in range(trial_n):
            
            if dataset_labelsc[i] in inter_label_type:
                self.inter_idx.append(i)
        self.inter_labelsc = dataset_labelsc[self.inter_idx]
        self.inter_label = dataset_label[self.inter_idx]
        # initialize shrinkage estimator
        self.mts_tar = MTS_estimator()
        self.mts_ntar = MTS_estimator()
        # get the index in trail of inter subclass
        all_target_idx = np.argwhere(dataset_label == 1).squeeze()
        target_inter_idx = [i for i in all_target_idx if dataset_labelsc[i] in inter_label_type]
        all_ntarget_idx = np.argwhere(dataset_label == 0).squeeze()
        ntarget_inter_idx = [i for i in all_ntarget_idx if dataset_labelsc[i] in inter_label_type]
        # downsample 
        ds_trainset = ave_downsample(trainset, ave_len)
        # fit get shrinkage parameter
        self.mts_tar.fit(ds_trainset[target_inter_idx, ...], dataset_labelsc[target_inter_idx], whitening=  whitening)
        self.tar_skmean2d, _, value1 = self.mts_tar.get_shrinkage_mean(trainset[target_inter_idx, ...])
        self.tar_skmean2d = np.concatenate((self.tar_skmean2d, trainset[dataset_label == 1,..., None].mean(axis = 0)), axis = 2)
        
        self.mts_ntar.fit(ds_trainset[ntarget_inter_idx, ...], dataset_labelsc[ntarget_inter_idx], whitening=  whitening)
        self.ntar_skmean2d, _, value2 = self.mts_ntar.get_shrinkage_mean(trainset[ntarget_inter_idx, ...])
        self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, trainset[dataset_label == 0,..., None].mean(axis = 0)), axis = 2)
        
        self.filter_set = np.zeros((chan_n, chan_n, self.cluster_n + 1))
        label_type = inter_label_type
        
        for i, ltype in enumerate(label_type):
            ltype = int(ltype)
            select_idx = np.argwhere(dataset_labelsc == ltype).squeeze()
            self.filter_set[..., i] = self.fit_class_all_subclass(trainset, dataset_label, select_idx , self.tar_skmean2d[..., i], self.ntar_skmean2d[..., i])                
        self.filter_set[..., self.cluster_n] = self.fit_class(np.transpose(trainset, (1, 2, 0)), trainset_label01)

        for cmp_idx in range(chan_n):
            for i in range(self.cluster_n):
                if self.filter_set[5:7,cmp_idx, i].sum() <=0:
                    self.filter_set[:,cmp_idx, i]  = self.filter_set[:,cmp_idx, i] * -1

        return self
    
    def _get_label_info(self, dataset_label, dataset_labelsc):
    
        target_label_type = np.unique(dataset_labelsc[dataset_label==1])
        ntarget_label_type = np.unique(dataset_labelsc[dataset_label==0])
        inter_label_type = np.intersect1d(target_label_type, ntarget_label_type)
        inter_label_type = inter_label_type 
        inter_idx = list()
        for i in range(dataset_labelsc.size):
            
            if dataset_labelsc[i] in inter_label_type:
                inter_idx.append(i)
        inter_labelsc = dataset_labelsc[self.inter_idx]
        inter_label = dataset_label[self.inter_idx]
        sub_class_type = self.inter_label_type.size
        main_class_type = np.unique(dataset_label)
        inter_labelsc = self.inter_labelsc.copy()
        
        y_list = list()
        for main_class in main_class_type:
            
            main_class_idx = np.argwhere(inter_label == main_class)
            
            y_class = np.zeros((inter_labelsc.size, self.cluster_n))
            inter_labelsc_main_class = inter_labelsc[main_class_idx]
            
            for i in range(inter_labelsc_main_class.size):
                
                idx = int(inter_labelsc_main_class[i])
                y_class[main_class_idx[i], idx] = 1
            
            y_list.append(y_class)
        y = np.hstack(y_list)
            
        return y, inter_idx, inter_label, inter_labelsc
        
    def _fit_MTS_DSP(self, trainset, dataset_label, dataset_labelsc, sub_class = 'intersect', whitening = False, ave_len = 1):
        
        self.mts = MTS_estimator()
        trial_n, chan_n, time_l = trainset.shape
        trainset_label = dataset_label.copy()
        label_type = np.unique(dataset_labelsc[trainset_label==1])
        self.cluster_n  = label_type.size
        self.mts.fit(trainset[trainset_label ==1, ...], dataset_labelsc[trainset_label ==1])
        self.ntar_2d = trainset[trainset_label==0,...].mean(axis = 0)
        self.tar_skmean2d, _, value = self.mts.get_shrinkage_mean(trainset[trainset_label ==1, ...])
        self.tar_skmean2d = np.concatenate((self.tar_skmean2d, trainset[trainset_label==1,..., None].mean(axis = 0)), axis = 2)
        self.ntar_skmean2d = self.ntar_2d
        self.filter_set = np.zeros((chan_n, chan_n, self.cluster_n + 1))
        
        for i, ltype in enumerate(label_type):
            ltype = int(ltype)
            select_idx_t = np.argwhere(dataset_labelsc == ltype).squeeze()
            select_idx_nt = np.argwhere(trainset_label == 0).squeeze()
            select_idx = np.concatenate((select_idx_t, select_idx_nt))
            subclass_data = trainset[select_idx, ...]
            trans_subclass_data = np.transpose(subclass_data, (1, 2, 0))
            subclass_label = trainset_label[select_idx]
            self.filter_set[..., i] = self.fit_class(trans_subclass_data, subclass_label, self.tar_skmean2d[..., i], self.ntar_2d[..., i])

        trans_subclass_data = np.transpose(trainset, (1, 2, 0))
        subclass_label = trainset_label    
        self.filter_set[..., self.cluster_n] = self.fit_class(np.transpose(trainset, (1, 2, 0)), trainset_label)


        for cmp_idx in range(chan_n):
            for i in range(self.cluster_n):
                if self.filter_set[5:7,cmp_idx, i].sum() <=0:
                    self.filter_set[:,cmp_idx, i]  = self.filter_set[:,cmp_idx, i] * -1
        return self
    
    def fit(self, trainseto, dataset_label, dataset_labelsc, cmp_num = 3, sub_class = 'intersect', whitening = False, ave_len = 1):
        
            
        if sub_class == 'intersect':
            
            self = self._fit_MTS_DSP_intersect(trainseto, dataset_label, dataset_labelsc, sub_class = 'intersect', whitening = False, ave_len = 1)
            
            # reshape data
            DSP_filter = self.filter_set
            tmp_num =  self.cluster_n +1
            ftrainset_list = list()
            rtrainset = np.transpose(trainseto[self.inter_idx,...], (1, 2, 0))
            for i in range(tmp_num):
                # get filtered data
                f_sc_trainset = self.transform(rtrainset[:,::,:], DSP_filter[:, :cmp_num, i], cmp_num)
                f_sc_trainset = f_sc_trainset - f_sc_trainset.mean(axis = (0, 1), keepdims = True)
                ftrainset_list.append(f_sc_trainset)

            ftrainset_3d = np.concatenate(ftrainset_list, axis = 1)
            rftrainset_3d =  np.transpose(ftrainset_3d, (2, 0, 1))
            trial_n, chan_n, time_l = rftrainset_3d.shape
            X = np.reshape(rftrainset_3d, (trial_n, time_l * chan_n) , order = "C")
            self.sub_class_type = self.inter_label_type.size
            main_class_type = np.unique(dataset_label)
            inter_labelsc = self.inter_labelsc.copy()
            
            y_list = list()
            for main_class in main_class_type:
                
                main_class_idx = np.argwhere(self.inter_label == main_class)
                
                y_class = np.zeros((trial_n, self.cluster_n))
                inter_labelsc_main_class = inter_labelsc[main_class_idx]
                
                for i in range(inter_labelsc_main_class.size):
                    
                    idx = int(inter_labelsc_main_class[i])
                    y_class[main_class_idx[i], idx] = 1
                
                y_list.append(y_class)
            self.y = np.hstack(y_list)

        
        else:
            
            self = self._fit_MTS_DSP(trainseto, dataset_label, dataset_labelsc, sub_class = 'intersect', whitening = False, ave_len = 1)
            # reshape data
            DSP_filter = self.filter_set
            tmp_num =  self.cluster_n +1
            ftrainset_list = list()
            rtrainset = np.transpose(trainset, (1, 2, 0))
            for i in range(tmp_num):
                # get filtered data
                f_sc_trainset = self.transform(rtrainset[:,::,:], DSP_filter[:, :cmp_num, i], cmp_num)
                f_sc_trainset = f_sc_trainset - f_sc_trainset.mean(axis = (0, 1), keepdims = True)
                ftrainset_list.append(f_sc_trainset)

            ftrainset_3d = np.concatenate(ftrainset_list, axis = 1)
            rftrainset_3d =  np.transpose(ftrainset_3d, (2, 0, 1))
            trial_n, chan_n, time_l = rftrainset_3d.shape
            X = np.reshape(rftrainset_3d, (trial_n, time_l * chan_n) , order = "C")
            
            self.y = np.zeros((trial_n, self.cluster_n + 1))
            for i in range(trial_n):
                idx = int(dataset_label[i])
                self.y[i, idx] = 1
            
        return self
  
    def fit_class(self, subclass_data, subclass_label, target_mean = None, nontarget_mean = None):
        
        trainset = subclass_data
        trainset_label = subclass_label
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        
        # data centralization
        for i in range(trainset.shape[2]):

            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
        trainset = train_trial
        # get class trial
        target = trainset[..., trainset_label.squeeze() == 1]
        nontarget = trainset[..., trainset_label.squeeze() == 0]
        # get class template
        if (target_mean is None) and (nontarget_mean is None):
            template_tar =  target.mean(axis = 2)       # extract target template
            template_nontar = nontarget.mean(axis = 2) # extract nontarget template
            # self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, template_nontar[..., None]), axis = 2)
            # self.tar_skmean2d = np.concatenate((self.tar_skmean2d, template_tar[..., None]), axis = 2)
            
        else:
            template_tar =  target_mean       # extract target template
            template_nontar = nontarget_mean # extract nontarget template
            
        template_all = (template_tar + template_nontar) / 2
        # calcute  between-class divergence matrix
        sigma = ((template_tar - template_all) @ (template_tar - template_all).T \
                + (template_nontar - template_all) @ (template_nontar - template_all).T)/2 
        Sb = sigma/time_len
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
        Sw_pre = (cov_0 + cov_1)/(2*time_len)
        
        ## calcute the shrinkage coefficient and shrinkage Sw
        
        if self.shrinkage is False:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            alpha = 0
        else:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            alpha, _ = shrink.oracle()
            alpha = alpha
       
        Sw = (1 - alpha) * Sw_pre + alpha * Tar
        
        # solve the optimizatino problem
        svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W = right_vector[:,denote_idx]
        # save DCPM model
        self.filter = np.real(sorted_W)
        self.target_tmp = template_tar
        self.nontarget_tmp = template_nontar
        return self.filter

    def fit_class_all_subclass(self, trainset, dataset_label, select_idx ,target_mean = None, nontarget_mean = None):
        
        trainset = np.transpose(trainset, (1, 2, 0))
        trainset_label = dataset_label
        chan_num, time_len, trial_num = trainset.shape
        train_trial = np.zeros(trainset.shape)
        
        # data centralization
        for i in range(trainset.shape[2]):

            train_trial[..., i] = trainset[:,:,i].squeeze() - np.mean(trainset[:,:,i], axis = 1, keepdims=True)
            
        # get class trial
        target = trainset[..., trainset_label.squeeze() == 1]
        nontarget = trainset[..., trainset_label.squeeze() == 0]
        # get class template
        if (target_mean is None) and (nontarget_mean is None):
            template_tar =  target.mean(axis = 2)       # extract target template
            template_nontar = nontarget.mean(axis = 2) # extract nontarget template
            # self.ntar_skmean2d = np.concatenate((self.ntar_skmean2d, template_nontar[..., None]), axis = 2)
            # self.tar_skmean2d = np.concatenate((self.tar_skmean2d, template_tar[..., None]), axis = 2)
            
        else:
            template_tar =  target_mean       # extract target template
            template_nontar = nontarget_mean # extract nontarget template
            
        template_all = (template_tar + template_nontar) / 2
        # calcute  between-class divergence matrix
        sigma = ((template_tar - template_all) @ (template_tar - template_all).T \
                + (template_nontar - template_all) @ (template_nontar - template_all).T)/2 
        Sb = sigma/time_len
        # calcute intra-class divergence matrix
        label_type = np.unique(dataset_label)
        
        cov_set = list()
        for type_idx, type_name in enumerate(label_type):
            idx = np.argwhere(trainset_label==type_name)
            cov_all = 0
            if type_name == 0:
                for n in range(idx.size):
                    cov_all += (trainset[..., idx[n]].squeeze() - template_nontar) \
                                @ (trainset[..., idx[n]].squeeze() - template_nontar).T
                cov_all = cov_all/(idx.size * time_len)
                cov_set.append(cov_all)  
            elif type_name !=0:
                for n in range(idx.size):
                    cov_all += (trainset[..., idx[n]].squeeze() - template_tar) \
                                        @ (trainset[..., idx[n]].squeeze() - template_tar).T    
                cov_all = cov_all/(idx.size * time_len)
                cov_set.append(cov_all)  
        cov_array = np.array(cov_set)
        Sw_pre = np.mean(cov_array, axis = 0)
        ## calcute the shrinkage coefficient and shrinkage Sw
        
        if self.shrinkage is False:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            alpha = 0
        else:
            P = Sw_pre.shape[1]
            F = np.trace(Sw_pre)/P
            Tar = F * (np.eye(Sw_pre.shape[0]))
            shrink = shrinkage_method(trainset, Sw_pre, Tar)
            alpha, _ = shrink.oracle()
            alpha = alpha
       
        Sw = (1 - alpha) * Sw_pre + alpha * Tar
        
        # solve the optimizatino problem
        svd_value , right_vector = scipy.linalg.eig(Sb, Sw)
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        sorted_V = svd_value[denote_idx]
        sorted_W = right_vector[:,denote_idx]
        # save DCPM model
        self.filter = np.real(sorted_W)
        self.target_tmp = template_tar
        self.nontarget_tmp = template_nontar
        return self.filter
  
    def transform(self, dataset, filter, cmp_num):
        """transform origin data to filtered data

        Args:
            dataset (ndarry): chan_num*time_len*trial_num
            cmp_num (int): the number of component used for filter

        Returns:
            ndarry: filtered datatset
        """
        _, time_len, trial_num = dataset.shape
        filtered_dataset = np.zeros((cmp_num, time_len, trial_num))
        W = filter[:,0:cmp_num]
        for i in range(trial_num):
            filtered_dataset[..., i] = W.T @ dataset[..., i] 
        return filtered_dataset
    
    def feature_extract(self, testset_o, cmp_num, ensamble = False):
        
        testset = np.transpose(testset_o, (1, 2, 0))
        # centralization
        location = np.mean(testset, axis = 1, keepdims = True)
        testset = testset - location
        trial_num = testset.shape[2]
        # extract model information
        template_tar = self.tar_skmean2d - self.tar_skmean2d.mean(axis = 1, keepdims=True)
        template_nontar =  self.ntar_skmean2d - self.ntar_skmean2d.mean(axis = 1, keepdims=True)
        # get filtered class template
        
        self.template_1 = list()
        self.template_0 = list()
        ftestset = list()
        DSP_filter = self.filter_set
        tmp_num =  self.cluster_n +1
        self.criterion = np.zeros((trial_num, tmp_num))
        self.predict_label = np.zeros((trial_num, tmp_num))
        
        self.feature = np.zeros((trial_num, (tmp_num)*cmp_num))
        self.feature1 = np.zeros((trial_num, cmp_num, tmp_num))
        
        if ensamble is True:
            
            self.ensamble_criterion = np.zeros((trial_num,))        
            self.ensamble_predict_label = np.zeros((trial_num,))
            # ensambel method
            ensamble_filter = DSP_filter[:, :cmp_num, 0]
            cat_template_tar = template_tar[..., 0]
            cat_template_nontar = template_nontar[..., 0]
            cat_testset = testset
            for i in range(tmp_num-1):
                ensamble_filter = np.concatenate((ensamble_filter, DSP_filter[:, :cmp_num, i+1]), axis = 1)
                cat_template_tar = np.concatenate((cat_template_tar, template_tar[..., i+1]), axis = 1)
                cat_template_nontar = np.concatenate((cat_template_nontar, template_nontar[..., i+1]), axis = 1)
                cat_testset = np.concatenate((cat_testset, testset), axis = 1)
                
            self.cat_template_1 = ensamble_filter.T @ cat_template_tar
            self.cat_template_0 = ensamble_filter.T @ cat_template_nontar
            ensambel_ftestset = self.transform(cat_testset, ensamble_filter, cmp_num*tmp_num)
        
            
            for j in range(trial_num):
                
                filtered_trial = ensambel_ftestset[..., j] 
                dist_ntar = np.linalg.norm(self.cat_template_0  - filtered_trial)**2
                dist_tar = np.linalg.norm(self.cat_template_1 - filtered_trial)**2
                    
                self.ensamble_criterion[j] = dist_ntar - dist_tar

            self.ensamble_predict_label = (np.sign(self.ensamble_criterion) + 1) / 2

            return self.ensamble_predict_label, self.ensamble_criterion
            
        else:
            
            for i in range(tmp_num):

                self.template_1.append(DSP_filter[:, :cmp_num, i].T @ template_tar[..., i])
                self.template_0.append(DSP_filter[:, :cmp_num, i].T @ template_nontar[...,i])
                # plt.plot(self.template_1[0].T)
                # plt.show()
                
                # get filtered data
                ftestset.append(self.transform(testset, DSP_filter[:, :cmp_num, i], cmp_num))
                
                # classification
                for j in range(trial_num):
                    
                    filtered_trial = ftestset[i][..., j] 
                    dist_ntar = np.linalg.norm(self.template_0[i] - filtered_trial)**2
                    dist_tar = np.linalg.norm(self.template_1[i] - filtered_trial)**2
                    self.criterion[j, i] = dist_ntar - dist_tar
                    self.feature[j, i*cmp_num : (i+1)*cmp_num] = self._discriminant_eigenvalue(filtered_trial, self.template_1[i], self.template_0[i])
         
                # statistic classification accuracy
                self.predict_label[:, i] = (np.sign(self.criterion[:, i]) + 1) / 2
            self.predict_label = self.predict_label.sum(axis = 1, keepdims = True)
            self.predict_label[self.predict_label>=3] = 1
            return self.feature, self.predict_label

    def reshape_data(self, trainset, cmp_num):
        rtrainset = np.transpose(trainset, (1, 2, 0))
        # reshape data
        DSP_filter = self.filter_set
        tmp_num =  self.cluster_n +1
        ftrainset_list = list()
        for i in range(tmp_num):
            # get filtered data
            f_sc_trainset = self.transform(rtrainset[:,::4,:], DSP_filter[:, :cmp_num, i], cmp_num)
            f_sc_trainset = f_sc_trainset - f_sc_trainset.mean(axis = (0, 1), keepdims = True)
            ftrainset_list.append(f_sc_trainset)

        ftrainset_3d = np.concatenate(ftrainset_list, axis = 1)
        rftrainset_3d =  np.transpose(ftrainset_3d, (2, 0, 1))
        trial_n, chan_n, time_l = rftrainset_3d.shape
        X = np.reshape(rftrainset_3d, (trial_n, time_l * chan_n) , order = "C")
        
        return X
    
    def predict(self, trainset, trainset_label, testset, cmp_num, ensamble = False):
        
        if ensamble is True:
            
            self.ensamble_label, self.ensamble_criterion = self.feature_extract(testset, cmp_num, ensamble = ensamble)
            return self.ensamble_label, self.ensamble_criterion
        else:
            
            trainset_label[trainset_label!=0] = 1
            train_feature, _= self.feature_extract(trainset, cmp_num)
            test_feature, plabel = self.feature_extract(testset, cmp_num)
            # fit srMTL
            self.srmtl = srMTL()
            self.srmtl.fit(train_feature, self.y)
            strain_feature = np.delete(train_feature, self.srmtl.select_idx, axis = 1)
            stest_feature = np.delete(test_feature, self.srmtl.select_idx, axis = 1)
            
            # preprocessing
            scaler =  MinMaxScaler()
            ztrain_eig_feature = scaler.fit_transform(strain_feature[:,:])
            ztest_eig_feature  = scaler.transform(stest_feature[:,:])

            # classification
            self.clf = SVC(
                C= 4.0, 
                kernel="linear", 
                degree= 3 , 
                gamma='scale', 
                coef0= 0.0001, 
                shrinking=True, 
                probability=False, 
                tol=0.001, 
                cache_size=1000, 
                class_weight= 'balanced', 
                verbose=False, 
                max_iter=1000, 
                decision_function_shape='ovo', 
                break_ties=False, 
                random_state=1079)

            self.clf.fit(ztrain_eig_feature, trainset_label)
            self.criterion = self.clf.decision_function(ztest_eig_feature)
            self.predict_label = self.clf.predict(ztest_eig_feature)
            # return self.predict_label, self.criterion
            return plabel, test_feature
        
    def score(self, trainset, trainset_label, testset, testset_label, cmp_num, TPlabel = 1, TNlabel = 0, ensamble = False):
        testset_label[testset_label != 0] = 1

        predict_label, criterion = self.predict(trainset, trainset_label, testset, cmp_num, ensamble = ensamble)
        result_1 = list()
        if ensamble is False:
            for i in range(predict_label.shape[1]):
                result_1.append(super().cal_score(testset_label, predict_label[:,i], criterion[:,i], TPlabel, TNlabel) )
            return result_1
        else:
            return super().cal_score(testset_label, predict_label, criterion, TPlabel, TNlabel)
#%%
fs = 250
time_win = [-0.15, 1]
time_segment = [0, 0.8]
file_path = r"H:\\6_汇报\\汇报21 基于聚类的子类学习方法\\dataset_AMUSE\\"
mat_name = r"sub_cali_{sub_idx}.mat"
#%%
# address
i = 30057
sub_idx = 5
# load data
mat_path = file_path + mat_name.format(sub_idx = sub_idx)
mat_data =  scipy.io.loadmat(mat_path)
slice_idx = slice(int(-time_win[0]*fs) + int(time_segment[0]*fs), int(time_segment[1]*fs) - int(time_win[0]*fs))
# extract data
dataset = mat_data['dataset'][:, :, slice_idx]
label01 = mat_data['dataset_01_label'].squeeze()
labelsc = mat_data['dataset_sbuclass_label'].squeeze()

# segment the trainset and Wtestset
trainset, testset, trainset_label01, testset_label01, trainset_labelsc, testset_labelsc = train_test_split(
    dataset, 
    label01, 
    labelsc,
    train_size = 0.7, 
    random_state = i, 
    shuffle = True
)

trainseto = np.transpose(trainset, (1, 2, 0))
testseto = np.transpose(testset, (1, 2, 0))
trainset_labelsc[trainset_labelsc==1] = 0
trainset_labelsc[trainset_labelsc==2] = 0
trainset_labelsc[trainset_labelsc==3] = 1
trainset_labelsc[trainset_labelsc==4] = 1
trainset_labelsc[trainset_labelsc==5] = 2
trainset_labelsc[trainset_labelsc==6] = 2

testset_labelsc[testset_labelsc==1] = 0
testset_labelsc[testset_labelsc==2] = 0
testset_labelsc[testset_labelsc==3] = 1
testset_labelsc[testset_labelsc==4] = 1
testset_labelsc[testset_labelsc==5] = 2
testset_labelsc[testset_labelsc==6] = 2

#%%
whitening = False
inter_chan = 1
mts_mtl_dcpm_c = MTS_MTL_DCPM_c(p=2, component_num=57)
mts_mtl_dcpm_c.fit(trainset[:,::inter_chan,:], trainset_label01, trainset_labelsc, sub_class = 'intersect', whitening = whitening, ave_len = 1)
metric1 = mts_mtl_dcpm_c.score(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], testset_label01, cmp_num = 14, ensamble = False)
ensamble_label, ensamble_criterion = mts_mtl_dcpm_c.predict(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], cmp_num = 10, ensamble = False)
print('mts_dcpm_2d')
print(metric1)

whitening = False
inter_chan = 1
mts_dcpm_c = MTS_DCPM_c(p=2, component_num=57)
mts_dcpm_c.fit(trainset[:,::inter_chan,:], trainset_label01, trainset_labelsc, sub_class = 'intersect', whitening = whitening, ave_len = 1)
metric1 = mts_dcpm_c.score(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], testset_label01, cmp_num = 14, ensamble = False)
ensamble_label, ensamble_criterion = mts_dcpm_c.predict(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], cmp_num = 10, ensamble = False)
print('mts_dcpm_2d')
print(metric1)
#%% multitarget shrinkage
ds_dataset = ave_downsample(dataset, 4)
mts = MTS_estimator()
mts.fit(ds_dataset[label01==1,...], labelsc[label01==1],whitening = False)
mean_list2, data2d_list2, value2 = mts.get_shrinkage_mean(dataset[label01==1,...])
#%%
whitening = False
inter_chan = 2
mts_dcpm = MTS_DCPM(p=2, component_num=57)
mts_dcpm.fit(trainset[:,::inter_chan,:], trainset_label01, trainset_labelsc, sub_class = 'intersect', whitening = whitening, ave_len = 1)
metric1 = mts_dcpm.score(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], testset_label01, cmp_num = 17, ensamble = False)
print('mts_dcpm_2d')
print(metric1)


whitening = False
inter_chan = 2
mts_dcpm_c = MTS_DCPM_c(p=2, component_num=57)
mts_dcpm_c.fit(trainset[:,::inter_chan,:], trainset_label01, trainset_labelsc, sub_class = 'intersect', whitening = whitening, ave_len = 1)
metric1 = mts_dcpm_c.score(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], testset_label01, cmp_num = 17, ensamble = False)
ensamble_label, ensamble_criterion = mts_dcpm_c.predict(trainset[:,::inter_chan,:], trainset_label01, testset[:,::inter_chan,:], cmp_num = 5, ensamble = False)
print('mts_dcpm_2d')
print(metric1)

#%%
inter_chan =2
dacie = DACIE(interval_coffe = 0, component_num = 10, p = 2)
dacie = dacie.fit2(trainseto[::inter_chan,...], trainset_label01)  
score2 = dacie.score(testseto[::inter_chan,...], testset_label01, cmp_num = 10, atype='discriminant eigenvalue')
print('dacie')
print(score2)

skdcpm = SKDCPM(component_num = 20)
skdcpm.fit(trainseto[::inter_chan,...], trainset_label01)
metric = skdcpm.score(testseto[::inter_chan,...], testset_label01, cmp_num = 5)
print('skdcpm')
print(metric)

dcpm = DCPM(component_num = 20)
dcpm.fit(trainseto[::inter_chan,...], trainset_label01)
metric = dcpm.score(testseto[::inter_chan,...], testset_label01, cmp_num = 5)
print('dcpm')
print(metric)
#%% multitarget shrinkage
ds_dataset = ave_downsample(dataset, 1)
mts = MTS_estimator()
mts.fit(ds_dataset[label01==1,...], labelsc[label01==1])
mean_list, data2d_list, value = mts.get_shrinkage_mean(dataset[label01==1,...])

ds_dataset = ave_downsample(dataset, 1)
mts = MTS_estimator()
mts.fit(ds_dataset[label01==0,...], labelsc[label01==0])
mean_list_nt, data2d_list_nt, value = mts.get_shrinkage_mean(dataset[label01==0,...])

#%%
chan_idx = 9
plt.plot(mean_list[chan_idx, :, 0], label = "origin")
plt.plot(data2d_list[chan_idx, :, 0], label = "2d")
plt.legend()
plt.show()

plt.plot(mean_list[chan_idx, :, 1],label = "origin")
plt.plot(data2d_list[chan_idx, :, 1],label = "2d")
plt.legend()
plt.show()

plt.plot(mean_list[chan_idx, :, 2],label = "origin")
plt.plot(data2d_list[chan_idx, :, 2],label = "2d")
plt.legend()
plt.show()
#%%
schan_idx = 3
chan_idx = 9
plt.figure()
for i in range(mean_list.shape[2]):

    plt.plot(mean_list[chan_idx, :, schan_idx], 'r')

for i in range(mean_list.shape[2]):

    plt.plot(mean_list_nt[chan_idx, :, schan_idx], 'b')
plt.ylim([-4, 4])
plt.show()
#%%
schan_idx = 5
chan_idx = 10
plt.figure()
for i in range(mean_list.shape[2]):

    plt.plot(mean_list[chan_idx, :, i], 'r')
    plt.plot(mean_list_nt[chan_idx, :, i], 'b')

plt.show()
#%%

plt.figure()
for i in range(mean_list.shape[2]):

    plt.plot(data2d_list[chan_idx, :, i])
plt.show()

# %%
chan_idx = 15
plt.plot(mean_list[chan_idx, :, 0] - mean_list[chan_idx, :, 1])
plt.show()
plt.plot(data2d_list[chan_idx, :, 0]-data2d_list[chan_idx, :, 1])
plt.show()
# %%
plt.plot(mean_list[chan_idx, :, 1] - mean_list[chan_idx, :, 2])
plt.show()
plt.plot(data2d_list[chan_idx, :, 1]-data2d_list[chan_idx, :, 2])
plt.show()

# %%
plt.plot(mean_list[chan_idx, :, 2] - mean_list[chan_idx, :, 0])
plt.show()
plt.plot(data2d_list[chan_idx, :, 2]-data2d_list[chan_idx, :, 0])
plt.show()

# %%
[27,32,63]
plt.plot(mean_list[:, :, 0].T)
plt.show()
plt.plot(mean_list[:, :, 1].T)
plt.show()
plt.plot(mean_list[:, :, 2].T)
plt.show()

# %%
plt.plot(mean_list[:, :, 0].T-mean_list[:, :, 1].T)
plt.show()
plt.plot(data2d_list[:, :, 0].T-data2d_list[:, :, 1].T)
plt.show()


plt.plot(mean_list[:, :, 0].T-mean_list[:, :, 2].T)
plt.show()
plt.plot(data2d_list[:, :, 0].T-data2d_list[:, :, 2].T)
plt.show()


plt.plot(mean_list[:, :, 2].T-mean_list[:, :, 1].T)
plt.show()
plt.plot(data2d_list[:, :, 2].T-data2d_list[:, :, 1].T)
plt.show()

# %%
