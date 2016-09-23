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
    mosaic with the distribution of given parameter
    measurement for each value of the independent
    variable.


"""

from __future__ import division


import glob
import logging
import os

import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

import pylab as pl

from scipy.stats import norm

import sklearn.cross_validation as cross


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
______ _       _   _   _               _   _ _   _ _ _ _
| ___ \ |     | | | | (_)             | | | | | (_) (_) | (_)
| |_/ / | ___ | |_| |_ _ _ __   __ _  | | | | |_ _| |_| |_ _  ___  ___
|  __/| |/ _ \| __| __| | '_ \ / _` | | | | | __| | | | __| |/ _ \/ __|
| |   | | (_) | |_| |_| | | | | (_| | | |_| | |_| | | | |_| |  __/\__ \
\_|   |_|\___/ \__|\__|_|_| |_|\__, |  \___/ \__|_|_|_|\__|_|\___||___/
                                __/ |
                               |___/
"""


def zrange(dir):
    files = glob.glob('{}/*.mfmtk'.format(dir))
    zrange = np.array([i.split('.mfmtk')[0].split('{}'.format(dir))[1] for i in
                      files])
    return sorted(zrange)


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
            ax.text(np.percentile(xlim, 5), np.percentile(ylim, 85),
                    'z = ' + str(xi))


def plot_as_gaussians(param, x, ax=None, color='blue', label=None, title=None,
                      ylabel=None, force=True):

    fit = np.array([np.zeros(2)]).reshape(2, 1)

    if ax is None:
        f, ax = plt.subplots(1, 1)

    xlim = [x.min(), x.max()]
    ax.set_xlim(xlim)

    for column in param.T:
        if(force):
            column[np.where(column == 0)] = ma.masked
            column_f = column[~column.mask]
        else:
            column_f = column

        mu, sigma = np.array(norm.fit(column_f.T))
        fit = np.append(fit, np.array([mu, sigma]).reshape(2, 1), axis=1)

    fit = fit.T[1:]

    if title is not None:
        ax.set_title(title, fontsize=20)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)

    ax.plot(x, fit.T[0], '-', color=color, label=label)
    ax.fill_between(x, fit.T[0] - fit.T[1], fit.T[0] + fit.T[1],
                    facecolor=color, alpha=0.5)


def adjust_ticks(ax):

    xticks = ax.get_xticks()
    ax.set_xticks(xticks[1:np.size(xticks) - 1])
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[1:np.size(yticks) - 1])

    return ax


def plot_2d_discriminant(data, classes, w, w0):
    plt.xlim(0.2, 1.2)
    plt.ylim(-0.5, -0.3)
    xx = np.linspace(-0.4, 1)
    a = -w[0] / w[1]
    yy = a * xx - w0 / w[1]
    fig = plt.subplot(111)
    plt.yticks([])
    plt.plot(xx, yy, '--k', lw=3,
             label=r"$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}  + w_0 = 0$")
    for c, marker, color in zip(classes, ('x', '^'), ('blue', 'red')):
        plt.scatter(x=data[:, 0].real[c],
                    y=data[:, 1].real[c],
                    marker=marker,
                    c=color,
                    alpha=0.5)
    plt.legend(loc=4)

"""
 _____ _               _  __ _           _   _               _   _ _   _ 
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
            if(zoo[1][np.where(zoo[0] == val)] == 'S'):
                classes[i] = 1
            elif(zoo[1][np.where(zoo[0] == val)] == 'E'):
                classes[i] = 2
    return classes


def classes_from_efigi(galaxies, t_type):
    classes = np.zeros(galaxies.shape, dtype='int8')
    ttype = t_type#[1].astype(float)

    e_indexes = galaxies[np.where(ttype <= 0)]#.T[0]
    s_indexes = galaxies[np.where(ttype > 0)]#.T[0]

    spirals = np.array([i for i, val in enumerate(galaxies)
                        if val in set(s_indexes)])
    logging.info(spirals)
    ellipticals = np.array([i for i, val in enumerate(galaxies)
                            if val in set(e_indexes)])
    classes[spirals] = 1
    classes[ellipticals] = 2
    return classes


def efigi_ttype(galaxies, t_type):
    ttype = np.zeros(galaxies.shape, dtype=float)
    for i, gal in enumerate(galaxies):
        if gal in t_type[0]:
            ttype[i] = t_type[1][np.where(t_type[0] == gal)[0][0]]
    return ttype


def classes_indexes(galaxies, t_type):
    ttype = t_type[1].astype(float)
    inf_lim = t_type[2].astype(float)
    sup_lim = t_type[3].astype(float)
    e_indexes = t_type.T[np.where(ttype <= 0)].T[0]
    s_indexes = t_type.T[np.where(ttype > 0)].T[0]
    spirals = np.array([i for i, val in enumerate(galaxies) if val in set(s_indexes)])
    ellipticals = np.array([i for i, val in enumerate(galaxies) if val in set(e_indexes)])
    return [spirals, ellipticals]


def find_class(galaxies, t_type, gclass):
    class_names = []
    for i, string_class in enumerate(t_type[2]):
        if gclass in string_class:
            class_names.append(t_type.T[i][0])

    indexes = np.array([i for i, val in enumerate(galaxies)
                       if val in set(class_names)])
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

def avaliador(classifi, X, Y, K=10):
    N = len(Y)
    kf = cross.KFold(n=N, n_folds=K)
    classifi.fit(X, Y)
    A = cross.cross_val_score(classifi, X, Y, cv=kf)
    P = cross.cross_val_score(classifi, X, Y, cv=kf, scoring='precision')
    R = cross.cross_val_score(classifi, X, Y, cv=kf, scoring='recall')
    F1 = cross.cross_val_score(classifi, X, Y, cv=kf, scoring='f1')
    # logging.info('A = {:6.2f}%'.format(A*100))
    # logging.info('P = {:6.2f}%'.format(P*100))
    # logging.info('R = {:6.2f}%'.format(R*100))
    # logging.info('F1= {:6.2f}%'.format(F1*100))

    return (A, P, R, F1)


def predict(classifier, x, y):
    predictions = cross.cross_val_predict(classifier, x, y, cv=10)
    return predictions


def train_discriminant(data, classes):

    # find mean vectors
    mean_vectors = []
    for c in classes:
        mean_vectors.append(np.mean(data.real[c], axis=0))
    mean_vectors

    # find covariance matrix
    sigma = np.cov(data[classes[0]], rowvar=0)

    # find prior probabilities
    ntot = data.real[classes[0]].shape[0] + data.real[classes[1]].shape[0]
    prior = []
    for c in classes:
        prior.append(data.real[c].shape[0] / ntot)

    sigma_i = np.linalg.inv(sigma)

    # find coefficients
    w = sigma_i.dot(mean_vectors[0] - mean_vectors[1])
    w0 = np.log(prior[0] / prior[1]) - 0.5 * ((mean_vectors[0]).T.dot(sigma_i)).dot(mean_vectors[0]) + 0.5 * ((mean_vectors[1]).T.dot(sigma_i)).dot(mean_vectors[1])

    return (w, w0)


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


def load_by_instrument(path, params=None, galaxies=None, classes=None):
    if(os.path.isdir(path)):
        instruments_cat = []
        for instrument in sorted(os.listdir(path)):
            list_cats = load_dir(path + instrument + '/', galaxies=galaxies, instrument=instrument, classes=classes)
            if((params is not None) and (galaxies is not None)):
                cats = reduce_by_params(list_cats, params=params, galaxies=galaxies)

                instruments_cat.append(cats)
            else:
                instruments_cat.append(list_cats)

        return np.array(instruments_cat)


def load_dir(path, galaxies=None, instrument=None, classes=None):
    if(os.path.isdir(path)):
        logging.info('Loading mfmtk catalog from path {}'.format(path))
        zs = zrange(path)
        catalogs = []
        for z in zs:
            cat = Catalog('{}{}.mfmtk'.format(path, z), zs=zs, galaxies=galaxies, instrument=instrument, classes=classes)
            cat.path = path
            catalogs.append(cat)
        return catalogs
    else:
        raise "Not a Directory"


def reduce_by_params(catalogs, params, galaxies=None, retro_reduce=True):
    new_catalogs = []
    for j, param in enumerate(params):
        reduced = catalogs[0].reduce(catalogs, param, galaxies=galaxies)
    
        if(retro_reduce):
           reduced.data = ma.mask_rows(reduced.data.T).T

    
        new_catalogs.append(reduced)

    

    if len(new_catalogs) < 2:
        return new_catalogs[0]



    return new_catalogs


def set_classes(catalogs, classes):

    for catalog in catalogs:
        catalog.set_classification(classes)

    return catalogs

def show_degradation(data, xlims=None, xticks=None, ylims=None, yticks=None, ylabels=None, titles=None):

    m, n = data.shape
    f, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))

    #plt.grid(True)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    bgcolor = ['#e3eeff', '#fff5e5', '#eaffd5', '#fffbbf']


    for i, column in enumerate(data):
        for j, cell in enumerate(column):
            k = (n * i + 1) + j - 1

            #logging.info('Entering Column {} and cell {}'.format(i+1, j+1))

            if(ylabels is not None):
                if(i == 0):
                    axes[j][i].set_ylabel(r'$\rm {}$'.format(ylabels[j]), fontsize=20)

            if(titles is not None):
                axes[0][i].set_title(r'$\rm {}$'.format(titles[i]), fontsize=20)


            axes[j][i].set_axis_bgcolor(bgcolor[i])

            if(i > 0):
                axes[j][i].set_yticklabels([])    

            if(j > n - 1):
                axes[j][i].set_xticklabels([])                 

            
            if(xticks is not None):
                axes[j][i].set_xticks(xticks[i])
            if(yticks is not None):
                axes[j][i].set_yticks(yticks[j])


            #axes[j][i].grid(True)
            
            if(ylims is not None):
                axes[j][i].set_ylim(ylims[j])

            __plot_degradation(cell, axes[j][i])

            if(xlims is not None):
                axes[j][i].set_xlim(xlims[i])

            #logging.info(axes[j][i].get_ylim())

    return f, axes

from astropy.stats import median_absolute_deviation as mad

def __plot_degradation(cat, ax, color='blue', label=None, title=None,
                       ylabel=None, force=True):
    
    cnames = ['Spirals', 'Ellipticals']
    colors = ['blue', 'red']
    linestyles = ['-', '-']
    for aclass, color, ls, cname in zip([1, 2], colors, linestyles, cnames):
        mus = []
        sigmas = []


        for column in cat.data:
            logging.info(np.median(column[np.where(cat.classes == aclass)].compressed().astype(float)))
            #mu, sigma = np.array(norm.fit(column[np.where(cat.classes == aclass)].compressed().astype(float)))
            mus.append(np.median(column[np.where(cat.classes == aclass)].compressed().astype(float)))
            sigmas.append(1.5*mad(column[np.where(cat.classes == aclass)].compressed().astype(float)))

        mus = np.array(mus)
        sigmas = np.array(sigmas)
        ax.plot(cat.zs.astype(float), mus, ls, lw=1.5, color=color, label=cname)
        #ax.plot(cat.zs.astype(float), mus-2*sigmas, '--', lw=1, color=color, label=label)
        #ax.plot(cat.zs.astype(float), mus+2*sigmas, '--', lw=1, color=color, label=label)
        ax.fill_between(cat.zs.astype(float), mus - sigmas, mus + sigmas, facecolor=color, alpha=0.5)
        #ax.fill_between(cat.zs.astype(float), mus - 2*sigmas, mus + 2*sigmas, facecolor=color, alpha=0.1)


def show_detections(cats):

    m, n = cats.shape
    for i in np.arange(0, m, 1):
        plt.plot(cats[i][0].zs, cats[i][0].detected_galaxies(), '-', label=cats[i][0].instrument)
        plt.legend()


class Catalog(object):
    """ The ''catalog'' class handles all Morfometryka
    catalogs reduction. It's implementation is not
    quite good, the majority of function could work
    standalone, but the 'self' keyword and the usage
    of the attribute ''reduced'' makes most of the
    reduction logic very straightforward.
    """

    def __init__(self, path='', reduced=False,
                 external=None, zs=None, galaxies=None,
                 classes=None, instrument=None):

        self.zs = np.array(zs)
        self.classes = classes
        self.instrument = instrument
        self.galaxies = galaxies
        self.path = path

        if(reduced):
            self.data = external
            self.reduced = reduced
        else:
            self.load(path)
            self.reduced = False

    def load(self, path):
            logging.info('Loading mfmtk catalog from file {}'.format(path))

            self.data = ma.asarray(np.loadtxt(path, delimiter=',',
                                   usecols=np.arange(1, len(column_dict), 1),
                                   dtype=float).T)

            self.__clean_data()

            galaxies = np.loadtxt(path, delimiter=',',
                                  usecols=[0], dtype='str').T

            for i, name in enumerate(galaxies):
                galaxies[i] = name.strip()

            self.galaxies = galaxies

    def __clean_data(self):

        if(np.isnan(self.data).sum() > 0):
            self.data[np.where(np.isnan(self.data))] = ma.masked

    def set_classification(self, classification):
        self.classes = classification

    def has_classes(self):
        return (self.classes is not None)

    def get_param(self, keyword):
        val = column_dict[keyword]
        print val
        return self.data[val]

    def get_z(self, z):
        z = str(z)
        if(self.reduced and len(np.where(self.zs == z)[0]) > 0):
            index = np.where(self.zs == z)[0][0]
            return self.data[index]
        else:
            raise Exception('Redshift section not found, try these \n {}'.
                            format(self.zs))

    def save(self, path, header):
        output = open(path, 'w')
        for headparam in header:
            output.write(headparam)
            output.write(',')
        output.write('\n')

        for line in self.data.T:
            for param in line:
                output.write(param)
                output.write(',')
            output.write('\n')

        output.close()

    def param_selection(self, params):
        new_catalog = np.array([])
        size = np.size(self.data[0])

        for i, param in enumerate(params):
            red_val = column_dict[param]

            if not i:
                new_catalog = self.data[red_val].reshape(size, 1)
            else:
                new_catalog = ma.append(new_catalog,
                                        self.data[red_val].reshape(size, 1),
                                        axis=1)

        return np.array(new_catalog).astype(float)

    def data(self):
        return self.data

    def reduce(self, others, reduce_column,
               masked=True, zs=None, galaxies=None):

        temp_cat = self
        # check if first element is self,
        # useful when passing a list with self in with
        if(self == others[0]):
            others = others[1:]

        for other in others:
            temp_cat = temp_cat.__reduce_masked(other, reduce_column, self.zs,
                                                galaxies=galaxies)
        return temp_cat

    def __reduce_masked(self, other, reduce_column, zs=None, galaxies=None):

        # use self galaxies if external list of galaxies is not provided
        # self.galaxies = galaxies
        red_val = column_dict[reduce_column]
        other_indexes = ma.array(np.zeros_like(galaxies))
        for i, galaxy in enumerate(galaxies):
            if galaxy in other.galaxies:
                other_indexes[i] = np.where(other.galaxies == galaxy)[0][0]
            else:
                other_indexes[i] = ma.masked

        new_catalog = ma.array([])
        if(self.reduced):
            columns = self.data.T
            new_catalog = ma.array(columns)
        else:
            column = ma.array(np.zeros_like(galaxies))
            for i, galaxy in enumerate(galaxies):
                if galaxy in self.galaxies:
                    index = np.where(self.galaxies == galaxy)[0][0]
                    column[i] = self.data[red_val][index]
                else:
                    column[i] = ma.masked

            column = column.reshape(column.shape[0], 1)
            new_catalog = ma.array(column)

        new_column = ma.array(np.zeros(galaxies.shape))
        for i, index in enumerate(other_indexes):
            if(ma.is_masked(index)):
                new_column[i] = ma.masked
            else:
                if(other.data[red_val][int(index)] is None):
                    new_column[i] = ma.masked
                else:
                    new_column[i] = other.data[red_val][int(index)]

        new_column = new_column.reshape(new_column.shape[0], 1)

        new_catalog = ma.append(new_catalog, new_column, axis=1)

        ncat = Catalog(path=self.path, reduced=True, external=new_catalog.T,
                       zs=zs, galaxies=galaxies, instrument=self.instrument, classes=self.classes)

        ncat.param = reduce_column

        return ncat

    def histogram_mosaic(self, color='b', bins=19, normed=1,
                         alpha=0.5, xinfo=False, xlim=None):
        """ Plots a mosaic with histograms for each value
        of 'x'.
        """

        if not self.reduced:
            raise Exception("This is not a reduced mfmtk catalog")

        if(self.zs is None):
            raise Exception("Redshift range not defined")

        size = len(self.zs)
        n = 3
        m = int(size / n)

        f, axes = plt.subplots(n, m, sharex=True, figsize=(15, 7))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, (zi, ax) in enumerate(zip(self.zs, axes.flat)):
            values = self.get_z(zi)

            if(self.has_classes()):
                classe1 = np.where(self.classes == 1)
                classe2 = np.where(self.classes == 2)
                val1 = values[classe1].compressed().astype(float)
                val2 = values[classe2].compressed().astype(float)

                ax.hist(val1, color=color,
                        bins=bins, normed=normed, alpha=alpha)
                ax.hist(val2, color='red',
                        bins=bins, normed=normed, alpha=alpha)
            else:
                ax.hist(values.compressed().astype(float), color=color,
                        bins=bins, normed=normed, alpha=alpha)

            ax.set_yticks([])
            #ax.set_xticks([])

            if(xlim is not None):
                ax.set_xlim(xlim)

    def detected_galaxies(self):
        gals = []   
        E = []
        S = []
        if(self.reduced):
            n = self.data.shape[0]
            for i in np.arange(0, n, 1):
                num_gals = len(self.data[i][~self.data[i].mask])
                gals.append(num_gals)        
                if(self.classes is not None):
                    S.append((self.classes[~self.data[i].mask] == 1).sum())
                    E.append((self.classes[~self.data[i].mask] == 2).sum())
        else:
            return len(self.data[0][~self.data[0].mask])

        return np.array(gals), np.array(E), np.array(S)



column_dict = { 'Mo' : 0,
                'No' : 1,
                'psffwhm' : 2,
                'asecpix' : 3,
                'skybg' : 4,
                'skybgstd' : 5,
                'x0peak' : 6,
                'y0peak' : 7,
                'x0col' : 8,
                'y0col' : 9,
                'x0A1fit' : 10,
                'y0A1fit' : 11,
                'x0A3fit' : 12,
                'y0A3fit' : 13,
                'a' : 14,
                'b' : 15,
                'PAdeg' : 16,
                'InFit1D' : 17,
                'RnFit1D' : 18,
                'nFit1D' : 19,
                'xsin' : 20,
                'x0Fit2D' : 21,
                'y0Fit2D' : 22,
                'InFit2D' : 23,
                'RnFit2D' : 24,
                'nFit2D' : 25,
                'qFit2D' : 26,
                'PAFit2D' : 27,
                'LT' : 28,
                'R10' : 29,
                'R20' : 30,
                'R30' : 31,
                'R40' : 32,
                'R50' : 33,
                'R60' : 34,
                'R70' : 35,
                'R80' : 36,
                'R90' : 37,
                'Rp' : 38,
                'C1' : 39,
                'C2' : 40,
                'A1' : 41,
                'A2' : 42,
                'A3' : 43,
                'A4' : 44,
                'S1' : 45,
                'S3' : 46,
                'G' : 47,
                'M20' : 48,
                'psi' : 49,
                'sigma_psi' : 50,
                'H' : 51,
                'QF' : 52
}
