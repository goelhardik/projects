import sys, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random

class LDA():
    def __init__(self, data, num_dims = 1, convert_data = 0, 
                 percentile = 50, threshold = 0, labelcol = -1, split_ratio = 0.9):
        '''
        Class constructor.
        --------------------------------
        data : the entire dataset
        num_dims : number of dimensions to project data into
        convert_data : flag to specify whether the data is to be converted to categorical
        percentile : if convert_data is True, then specify the percentile for conversion
        threshold : flag to indicate whether to do thresholding or gaussian modeling for classification
        labelcol : which column in the csv data contains the label
        split_ratio : split ratio for train-test split
        '''
        self.data = data
        self.num_dims = num_dims
        self.convert_data = convert_data
        self.percentile = percentile
        self.threshold = threshold
        self.labelcol = labelcol
        self.split_ratio = split_ratio
        if (self.convert_data):
            self.data = self.to_categorical(self.data, self.percentile)

    '''
    Function to convert data to categorical.
    To be used only for the boston dataset.
    '''
    def to_categorical(self, data, percentile):
        fraction = percentile / 100.0

        # partition data based on percentile of label column
        med = data.ix[:, self.labelcol].quantile(fraction)
        for i in range(data.shape[0]):
            if (data.ix[i, self.labelcol] >= med):
                data.ix[i, self.labelcol] = 1 
            else:
                data.ix[i, self.labelcol] = 0 

        return data

    '''
    Utility function to drop some column from the given pandas dataframe.
    '''
    def drop_col(self, data, col):
        return data.drop(data.columns[[col]], axis = 1)

    '''
    Main function to apply LDA
    '''
    def fit(self):
        # Function estimates the LDA parameters
        def estimate_params(data):
            # group data by label column
            grouped = data.groupby(self.data.ix[:,self.labelcol])

            # calculate means for each class
            means = {}
            for c in self.classes:
                means[c] = np.array(self.drop_col(self.classwise[c], self.labelcol).mean(axis = 0))

            # calculate the overall mean of all the data
            overall_mean = np.array(self.drop_col(data, self.labelcol).mean(axis = 0))

            # calculate between class covariance matrix
            # S_B = \sigma{N_i (m_i - m) (m_i - m).T}
            S_B = np.zeros((data.shape[1] - 1, data.shape[1] - 1))
            for c in means.keys():
                S_B += np.multiply(len(self.classwise[c]),
                                   np.outer((means[c] - overall_mean), 
                                            (means[c] - overall_mean)))

            # calculate within class covariance matrix
            # S_W = \sigma{S_i}
            # S_i = \sigma{(x - m_i) (x - m_i).T}
            S_W = np.zeros(S_B.shape) 
            for c in self.classes: 
                tmp = np.subtract(self.drop_col(self.classwise[c], self.labelcol).T, np.expand_dims(means[c], axis=1))
                S_W = np.add(np.dot(tmp, tmp.T), S_W)

            # objective : find eigenvalue, eigenvector pairs for inv(S_W).S_B
            mat = np.dot(np.linalg.pinv(S_W), S_B)
            eigvals, eigvecs = np.linalg.eig(mat)
            eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

            # sort the eigvals in decreasing order
            eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

            # take the first num_dims eigvectors
            w = np.array([eiglist[i][1] for i in range(self.num_dims)])

            self.w = w
            self.means = means
            return


        # perform train-test split
        traindata = []
        testdata = []
        # group data by label column
        grouped = data.groupby(self.data.ix[:,self.labelcol])
        self.classes = [c for c in grouped.groups.keys()]
        self.classwise = {} 
        for c in self.classes:
            self.classwise[c] = grouped.get_group(c)
            rows = random.sample(self.classwise[c].index, 
                                     int(self.classwise[c].shape[0] * 
                                     self.split_ratio))
            traindata.append(self.classwise[c].ix[rows])
            testdata.append(self.classwise[c].drop(rows))

        traindata = pd.concat(traindata)
        testdata = pd.concat(testdata)

        # estimate the LDA parameters
        estimate_params(traindata)
        # perform classification on test set
        # if the method is threshold
        if (self.threshold):
            self.calculate_threshold()
            # append the training and test error rates for this iteration
            trainerror = self.calculate_score(traindata) / float(traindata.shape[0])
                                                                
            testerror = self.calculate_score(testdata) / float(testdata.shape[0])

        # if the method is gaussian modeling
        else:
            self.gaussian_modeling()
            # append the training and test error rates for this iteration
            trainerror = self.calculate_score_gaussian(traindata) / float(traindata.shape[0])
            testerror = self.calculate_score_gaussian(testdata) / float(testdata.shape[0])

        return trainerror, testerror

    '''
    Function to calculate the classification threshold.
    Projects the means of the classes and takes their mean as the threshold.
    Also specifies whether values greater than the threshold fall into class 1 
    or class 2.
    '''
    def calculate_threshold(self):
        # project the means and take their mean
        tot = 0
        for c in self.means.keys():
            tot += np.dot(self.w, self.means[c])
        self.w0 = 0.5 * tot

        # for 2 classes case; mark if class 1 is >= w0 or < w0
        c1 = self.means.keys()[0]
        c2 = self.means.keys()[1]
        mu1 = np.dot(self.w, self.means[c1])
        if (mu1 >= self.w0):
            self.c1 = 'ge'
        else:
            self.c1 = 'l'

    '''
    Function to calculate the scores in thresholding method.
    Assigns predictions based on the calculated threshold.
    '''
    def calculate_score(self, data):
        inputs = self.drop_col(data, self.labelcol)
        # project the inputs
        proj = np.dot(self.w, inputs.T).T
        # assign the predicted class
        c1 = self.means.keys()[0]
        c2 = self.means.keys()[1]
        if (self.c1 == 'ge'):
            proj = [c1 if proj[i] >= self.w0 else c2 for i in range(len(proj))]
        else:
            proj = [c1 if proj[i] < self.w0 else c2 for i in range(len(proj))]
        # calculate the number of errors made
        errors = (proj != data.ix[:, self.labelcol])
        return sum(errors)

    '''
    Function to estimate gaussian models for each class.
    Estimates priors, means and covariances for each class.
    '''
    def gaussian_modeling(self):
        self.priors = {}
        self.gaussian_means = {}
        self.gaussian_cov = {}

        for c in self.means.keys():
            inputs = self.drop_col(self.classwise[c], self.labelcol)
            proj = np.dot(self.w, inputs.T).T
            self.priors[c] = inputs.shape[0] / float(self.data.shape[0])
            self.gaussian_means[c] = np.mean(proj, axis = 0)
            self.gaussian_cov[c] = np.cov(proj, rowvar=False)

    '''
    Utility function to return the probability density for a gaussian, given an 
    input point, gaussian mean and covariance.
    '''
    def pdf(self, point, mean, cov):
        cons = 1./((2*np.pi)**(len(point)/2.)*np.linalg.det(cov)**(-0.5))
        return cons*np.exp(-np.dot(np.dot((point-mean),np.linalg.inv(cov)),(point-mean).T)/2.)

    '''
    Function to calculate error rates based on gaussian modeling.
    '''
    def calculate_score_gaussian(self, data):
        classes = sorted(list(self.means.keys()))
        inputs = self.drop_col(data, self.labelcol)
        # project the inputs
        proj = np.dot(self.w, inputs.T).T
        # calculate the likelihoods for each class based on the gaussian models
        likelihoods = np.array([[self.priors[c] * self.pdf([x[ind] for ind in 
                                                            range(len(x))], self.gaussian_means[c], 
                               self.gaussian_cov[c]) for c in 
                        classes] for x in proj])
        # assign prediction labels based on the highest probability
        labels = np.argmax(likelihoods, axis = 1)
        errors = np.sum(labels != data.ix[:, self.labelcol])
        return errors

    def plot_bivariate_gaussians(self):
        classes = list(self.means.keys())
        colors = cm.rainbow(np.linspace(0, 1, len(classes)))
        plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

        fig = plt.figure()
        ax3D = fig.add_subplot(111, projection='3d')
        for c in self.means.keys():
            data = np.random.multivariate_normal(self.gaussian_means[c], 
                                                 self.gaussian_cov[c], size=100)
            pdf = np.zeros(data.shape[0])
            cons = 1./((2*np.pi)**(data.shape[1]/2.)*np.linalg.det(self.gaussian_cov[c])**(-0.5))
            X, Y = np.meshgrid(data.T[0], data.T[1])
            def pdf(point):
                return cons*np.exp(-np.dot(np.dot((point-self.gaussian_means[c]),np.linalg.inv(self.gaussian_cov[c])),(point-self.gaussian_means[c]).T)/2.)

            zs = np.array([pdf(np.array(ponit)) for ponit in zip(np.ravel(X), 
                                                                   np.ravel(Y))])
            Z = zs.reshape(X.shape)
            surf = ax3D.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                                       color=plotlabels[c], linewidth=0, 
                                       antialiased=False)
        plt.show()

    def plot_proj_1D(self, data):
        classes = list(self.means.keys())
        colors = cm.rainbow(np.linspace(0, 1, len(classes)))
        plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

        fig = plt.figure()
        for i, row in data.iterrows():
            proj = np.dot(self.w, row[:self.labelcol])
            plt.scatter(proj, np.random.normal(0,1,1)+0, color = 
                        plotlabels[row[self.labelcol]])
        plt.show()

    def plot_proj_2D(self, data):
        classes = list(self.means.keys())
        colors = cm.rainbow(np.linspace(0, 1, len(classes)))
        plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

        fig = plt.figure()
        for i, row in data.iterrows():
            proj = np.dot(self.w, row[:self.labelcol])
            plt.scatter(proj[0], proj[1], color = 
                        plotlabels[row[self.labelcol]])
        plt.show()
    
if __name__ == '__main__':
    data = pd.read_csv(sys.argv[1])
    labelcol = int(sys.argv[2])
    lda = LDA(data, num_dims=2, convert_data=0, threshold=0, labelcol=labelcol)
    trainerror, testerror = lda.fit()
    print(trainerror)
    print(testerror)
    #print(verifyLDA(data, data, labelcol))
    lda.plot_proj_2D(data)
    lda.plot_bivariate_gaussians()
