# -*- coding: utf-8 -*-
""" mfmtk-utils: Morfometryka Utilities

This module is a collection of classes and functions created
to help in the reduction of MFMTK's data. There's three
categories of utilities: general, reduction, and plotting. 

1. General 
    * Functions to help in general problems
2. Reduction
    * Classes and methods to reduce MFMTK catalogs
    in desirable outputs.

Example: 
    The method ''reduce_masked'' from the
    ''catalog'' class takes a list of MFMTK catalogs
    and a MFMTK's parameter as argument making then 
    a combination from all catalogs in the list for the
    parameter passed as argument. The result is a catalog
    with 'n+1' columns where 'n' is the number of
    catalogs in the argument list plus a column with
    all galaxies names as in the first catalog. 

3. Plotting
    * Plotting routines were created to facilitate the
    creation of some common plots under MFMTK's workflow.

Example:
    The ''histogram'' function, for example, creates a 
    mosaic with the distribuction of given parameter
    measurement for each value of the independent 
    variable.


"""
from __future__ import division
from scipy.stats import norm
import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
LOG_FILENAME = '/data/mfmtkutils.log'
logging.basicConfig(filename=LOG_FILENAME)

def intersect(a, b):
    """ Return an numpy array with the intersection
        of a and b.
    """
    return np.array(list(set(a) & set(b)))


"""
______ _       _   _   _               _   _ _   _ _ _ _   _        
| ___ \ |     | | | | (_)             | | | | | (_) (_) | (_)          
| |_/ / | ___ | |_| |_ _ _ __   __ _  | | | | |_ _| |_| |_ _  ___  ___ 
|  __/| |/ _ \| __| __| | '_ \ / _` | | | | | __| | | | __| |/ _ \/ __|
| |   | | (_) | |_| |_| | | | | (_| | | |_| | |_| | | | |_| |  __/\__ \
\_|   |_|\___/ \__|\__|_|_| |_|\__, |  \___/ \__|_|_|_|\__|_|\___||___/
                                __/ |                                  
                               |___/                                   
"""

def histograms(param, x, axes=None, color='b', bins=19, normed=1,
               alpha=0.5, xinfo=False):
    """ Plots a mosaic with histograms for each value
    of 'x'.
    """
    if axes is None:
        f, axes = plt.subplots(4, 5, sharex=True, figsize=(15, 7))
        plt.subplots_adjust(hspace=0, wspace=0)

    for i, (column, ax, xi) in enumerate(zip(param.T, axes.flat, x)):
        ax.set_yticks([])
        ax.hist(column, color=color,
         bins=bins, normed=normed, alpha=alpha)
        if(xinfo):
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.text(np.percentile(xlim, 5), np.percentile(ylim, 85), 'z = ' + str(xi))
    
def plot_as_gaussians(param, x, ax=None, color='blue', label=None, title=None, ylabel=None):
    fit = np.array([np.zeros(2)]).reshape(2,1)

    if ax is None:
        f, ax = plt.subplots(1, 1)

    xlim=[x.min(), x.max()]
    ax.set_xlim(xlim)

    for column in param.T:
        mu, sigma = np.array(norm.fit(column.T))
        fit = np.append(fit, np.array([mu, sigma]).reshape(2,1), axis=1)

    fit = fit.T[1:]

    if title is not None:
        ax.set_title(title, fontsize=20)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)

    ax.plot(x, fit.T[0], '-', color=color, label=label)
    ax.fill_between(x, fit.T[0] - fit.T[1], fit.T[0] + fit.T[1],
                     facecolor=color, alpha=0.5)

def plot_2Ddiscriminant(data, classes, W, w0):
    plt.xlim(0.2, 1.2)
    plt.ylim(-0.5, -0.3)
    xx = np.linspace(-0.4, 1) 
    a = -W[0] / W[1] 
    yy = a * xx - w0 / W[1]
    #f, ax = plt.subplots(1,1, figsize=(7, 6))
    fig = plt.subplot(111)
    plt.yticks([])
    plt.plot(xx,yy, '--k', lw=3, label=r"$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}  + w_0 = 0$")
    for c, marker, color in zip(classes, ('x', '^'), ('blue', 'red')):
        plt.scatter(x=data[:,0].real[c],
                y=data[:,1].real[c],
                marker=marker,
                c=color,
                alpha=0.5)
    plt.legend(loc=4)

"""
 _____ _               _  __ _           _   _               _   _ _   _ _     
/  __ \ |             (_)/ _(_)         | | (_)             | | | | | (_) |    
| /  \/ | __ _ ___ ___ _| |_ _  ___ __ _| |_ _  ___  _ __   | | | | |_ _| |___ 
| |   | |/ _` / __/ __| |  _| |/ __/ _` | __| |/ _ \| '_ \  | | | | __| | / __|
| \__/\ | (_| \__ \__ \ | | | | (_| (_| | |_| | (_) | | | | | |_| | |_| | \__ \
 \____/_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|  \___/ \__|_|_|___/
                                                                               
"""                                                                            

def classes_from_zoo(galaxies, zoo):
    classes = np.zeros(galaxies.shape, dtype='int8') 
    for i, val in enumerate(galaxies):
        if(val in zoo[0]):
            #logging.info('galaxy {} is type {}'.format(val, zoo[0][np.where(zoo[0] == val)]))
            if(zoo[1][np.where(zoo[0] == val)] == 'S'):
                classes[i] = 1
            elif(zoo[1][np.where(zoo[0] == val)] == 'E'):
                classes[i] = 2
    return classes

def classes_from_EFIGI(galaxies, T_type):
    classes = np.zeros(galaxies.shape, dtype='int8') 
    ttype = T_type[1].astype(float)
    E_indexes = T_type.T[np.where(ttype <= 0)].T[0]
    S_indexes = T_type.T[np.where(ttype > 0)].T[0]
    spirals = np.array([i for i, val in enumerate(galaxies) if val in set(S_indexes)])
    ellipticals = np.array([i for i, val in enumerate(galaxies) if val in set(E_indexes)])
    classes[spirals] = 1
    classes[ellipticals] = 2
    return classes

def EFIGI_ttype(galaxies, T_type):
    ttype = np.zeros(galaxies.shape, dtype=float) 
    for i, gal in enumerate(galaxies):
        if gal in T_type[0]:
            ttype[i] = T_type[1][np.where(T_type[0] == gal)[0][0]]
    return ttype

def classes_indexes(galaxies, T_type):
    ttype = T_type[1].astype(float)
    inf_lim = T_type[2].astype(float)
    sup_lim = T_type[3].astype(float)
    E_indexes = T_type.T[np.where(ttype <= 0)].T[0]
    S_indexes = T_type.T[np.where(ttype > 0)].T[0]
    spirals = np.array([i for i, val in enumerate(galaxies) if val in set(S_indexes)])
    ellipticals = np.array([i for i, val in enumerate(galaxies) if val in set(E_indexes)])
    return [spirals, ellipticals]

def find_class(galaxies, T_type, gclass):
    class_names = []
    for i, string_class in enumerate(T_type[2]):
        if gclass in string_class:
            class_names.append(T_type.T[i][0])

    indexes = np.array([i for i, val in enumerate(galaxies) if val in set(class_names)])
    return indexes

def fisher_lda(X, n, classes):
    
    #find mean vectors for each class
    mean_vectors = []
    for c in classes:
        mean_vectors.append(np.mean(X[c], axis=0))
    
    #overall mean for the data
    overall_mean = np.mean(X, axis=0)   

    #find  within-class scatter matrix
    SW = np.zeros((n,n))
    for c in classes:
        SW_temp = np.zeros((n,n))
        for galaxy in X[c]:
            galaxy, mv = galaxy.reshape(n, 1), mean_vectors[0].reshape(n,1)
            SW_temp += (galaxy-mv).dot((galaxy-mv).T)
        SW = SW + SW_temp

    #find between-class scatter matrix
    S_B = np.zeros((n,n))
    for galclass, mv in zip(classes, mean_vectors):
        N = X[galclass].shape[0]
        mv = mv.reshape(n,1)
        overall_mean = overall_mean.reshape(n,1)
        S_B  += N* (mv - overall_mean).dot((mv-overall_mean).T)
        

    #solve the eigenvalue problem for our matrixes
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(SW).dot(S_B))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    
    #take only the most significant LDAs
    W = np.hstack((eig_pairs[0][1].reshape(5,1), eig_pairs[1][1].reshape(5,1)))

    #transform our data in the new subspace
    X_lda = X.dot(W)

    return X_lda

import pylab as pl
def lda_report_normalize(lda, data):
    logging.info('Normalizing coefficients and calculating Mi')
    wn  = -lda.coef_[0]/pl.norm(lda.coef_)
    w0n = -lda.intercept_[0]/pl.norm(lda.coef_)
    Mi   = np.dot(wn, data.T) + w0n
    logging.info('w    = {}'.format(lda.coef_[0]))
    logging.info('w0   = {}'.format(lda.intercept_[0]))
    logging.info('w~   = {}'.format(wn))
    logging.info('w0/w = {}'.format(w0n))
    return Mi

import sklearn.cross_validation as cross

def avaliador(classifi, X, Y, K=10):
    N  = len(Y)
    kf = cross.KFold(n=N, n_folds=K)
    classifi.fit(X, Y)
    A = np.mean(cross.cross_val_score(classifi, X, Y, cv=kf))
    P = np.mean(cross.cross_val_score(classifi, X, Y, cv=kf, scoring='precision'))
    R = np.mean(cross.cross_val_score(classifi, X, Y, cv=kf, scoring='recall'))
    F1 = np.mean(cross.cross_val_score(classifi, X, Y, cv=kf, scoring='f1'))
    logging.info('A = {:6.2f}%'.format(A*100))
    logging.info('P = {:6.2f}%'.format(P*100))
    logging.info('R = {:6.2f}%'.format(R*100))
    logging.info('F1= {:6.2f}%'.format(F1*100))

    return (A, P, R, F1)

def predict(classifier, X, Y):
    predictions = cross.cross_val_predict(classifier, X, Y, cv=10)
    return predictions

def train_discriminant(data, classes):
    n = 5
    #find mean vectors
    mean_vectors = []
    for c in classes:
        mean_vectors.append(np.mean(data.real[c], axis=0))
    mean_vectors

    #find covariance matrix
    SIGMA = np.cov(data[classes[0]], rowvar=0)

    #find prior probabilities
    Ntot = data.real[classes[0]].shape[0] + data.real[classes[1]].shape[0]
    prior = []
    for c in classes:
        prior.append(data.real[c].shape[0]/Ntot)

    SIGMA_I = np.linalg.inv(SIGMA)

    #find coefficients
    W = SIGMA_I.dot(mean_vectors[0] - mean_vectors[1])
    w0 = np.log(prior[0]/prior[1]) - 0.5 * ((mean_vectors[0]).T.dot(SIGMA_I)).dot(mean_vectors[0]) + 0.5 * ((mean_vectors[1]).T.dot(SIGMA_I)).dot(mean_vectors[1])

    return (W, w0)


def find_spiked(param, threshold):
    spiked = []
    not_spiked = []
    for i, galaxy in enumerate(Rn):
        derivative = np.gradient(galaxy)
        if not (np.size(derivative[np.where(derivative > 0)]) >= threshold):
            not_spiked.append(i)
        else:
            spiked.append(i)
    return (np.array(spiked), np.array(not_spiked))

class catalog(object):
    """ The ''catalog'' class handles all Morfometryka
    catalogs reduction. It's implementation is not
    quite good, the majority of function could work
    standalone, but the 'self' keyword and the usage
    of the attribute ''reduced'' makes most of the
    reduction logic very straightforward.
    """
    
    def __init__(self, path='', reduced=False, external=None):
        if(reduced):
            self.raw_catalog = external
            self.reduced = reduced
        else:
            self.load(path)
            self.reduced = False    
    
    def load(self, path):
        cat = np.loadtxt(path, delimiter=',', dtype='str').T
        for i, name in enumerate(cat[0]):
            cat[0][i] = name.strip()
        self.raw_catalog = cat
    
    def save(self, path, header):
        output = open(path, 'w')
        for headparam in header:
            output.write(headparam)
            output.write(',')
        output.write('\n')
            
        for line in self.raw_catalog.T:
            for param in line:
                output.write(param)
                output.write(',')
            output.write('\n')
            
        output.close()     
    
    def param_selection(self, params):
        new_catalog = np.array([])
        size = np.size(self.raw_catalog[0])

        for i, param in enumerate(params):
            red_val = column_dict[param]
            
            if not i:
                new_catalog = self.raw_catalog[red_val].reshape(size, 1)
            else:
                new_catalog = ma.append(new_catalog, self.raw_catalog[red_val].reshape(size, 1), axis=1)    
        
        return np.array(new_catalog).astype(float)

    def data():
        return self.raw_catalog

    def reduce(self, others, reduce_column, masked=True):
        temp_cat = self

        #check if first element is self, useful when passing a list with self in with
        if(self == others[0]):
            others = others[1:]

        for other in others:
            temp_cat = temp_cat.__reduce_masked(other, reduce_column)
            
        return temp_cat    

    def __reduce_masked(self, other, reduce_column):
        red_val = column_dict[reduce_column]
        galaxies = self.raw_catalog[0]
        other_indexes = ma.array(np.zeros_like(galaxies))

        for i, galaxy in enumerate(galaxies):
            if galaxy in other.raw_catalog[0]:
                other_indexes[i] = np.where(other.raw_catalog[0] == galaxy)[0][0]


        new_catalog = ma.array([])
        galaxies = galaxies.reshape(galaxies.shape[0], 1)

        if(self.reduced):
            columns = self.raw_catalog[1:].T
            new_catalog = ma.append(galaxies, columns, axis=1)
        else:
            column =  self.raw_catalog[red_val].reshape(self.raw_catalog[red_val].shape[0], 1)
            new_catalog = ma.append(galaxies, column, axis=1)
        

        new_column = ma.array(np.zeros(galaxies.shape))
        other_indexes[np.where(other_indexes == '')] = 0

        for i, index in enumerate(other_indexes):
            if(index == 0):
                new_column[i] = ma.masked
            else:
                new_column[i] = other.raw_catalog[red_val][int(index)]

        new_catalog = ma.append(new_catalog, new_column, axis=1)

        ncat = catalog(reduced=True, external=new_catalog.T)
        return ncat



    
        
        
        
column_dict = {'galaxy_name': 0, 
    'Mo': 1, 
    'No': 2, 
    'psffwh': 3, 
    'asecpix': 4, 
    'skybg': 5, 
    'skybgstd': 6, 
    'x0peak': 7, 
    'y0peak': 8, 
    'x0col': 9, 
    'y0col': 10, 
    'x0A1fit': 11, 
    'y0A1fit': 12, 
    'x0A3fit': 13, 
    'y0A3fit': 14, 
    'a': 15, 
    'b': 16, 
    'PAdeg': 17, 
    'InFit1D': 18, 
    'RnFit1D': 19, 
    'nFit1D': 20, 
    'xsin': 21, 
    'x0Fit2D': 22, 
    'y0Fit2D': 23, 
    'InFit2D': 24, 
    'RnFit2D': 25, 
    'nFit2D': 26, 
    'qFit2D': 27, 
    'PAFit2D': 28, 
    'Rp': 29, 
    'C1': 30, 
    'C2': 31, 
    'R20': 32, 
    'R50': 33, 
    'R80': 34, 
    'R90': 35, 
    'A1': 36, 
    'A2': 37, 
    'A3': 38, 
    'A4': 39, 
    'S1': 40, 
    'S3': 41, 
    'G': 42, 
    'M20': 43, 
    'psi': 44, 
    'sigma_psi': 45, 
    'H': 46, 
    'QF' : 47
    }