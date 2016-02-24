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

import numpy as np
import numpy.ma as ma 

from scipy.stats import norm


def intersect(a, b):
    """ Return an numpy array with the intersection
        of a and b.
    """
    return np.array(list(set(a) & set(b)))

def histograms(param, x, axes, color='b', bins=25, normed=1, alpha=0.5):
    """ Plots a mosaic with histograms for each value
    of 'x'.
    """
    for column, ax, xi in zip(param.T, axes.flat, x):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.hist(column, color=color,
         bins=bins, normed=normed, alpha=alpha)
    
def plot_as_gaussians(param, x, ax, color='blue', labels=None):
    fit = np.array([np.zeros(2)]).reshape(2,1)

    for column in param.T:
        mu, sigma = np.array(norm.fit(column.T))
        fit = np.append(fit, np.array([mu, sigma]).reshape(2,1), axis=1)

    fit = fit.T[1:]

    ax.plot(x, fit.T[0], '-', color=color, label=label)
    ax.fill_between(x, fit.T[0] - fit.T[1], fit.T[0] + fit.T[1],
                     facecolor=color, alpha=0.5)

   


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