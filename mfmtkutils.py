import numpy as np
import numpy.ma as ma 


def histograms(param, x, axes, color='b', bins=25, normed=1, alpha=0.5):
    for column, ax, xi in zip(param.T, axes.flat, x):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.hist(column, color=color,
         bins=bins, normed=normed, alpha=alpha)
    


class catalog(object):
    
    
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
    
        
    def reduce_catalogs(self, others, reduce_column, wout_QF=False):
        temp_cat = self
        for other in others:
            temp_cat = temp_cat.reduce_catalog(other, reduce_column, wout_QF)
            
        return temp_cat
    
    def reduce_cats_masked(self, others, reduce_column):
        temp_cat = self
        for other in others:
            temp_cat = temp_cat.reduce_masked(other, reduce_column)
            
        return temp_cat    

    def reduce_masked(self, other, reduce_column):

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

        ncat = catalog(new_catalog.T)
        ncat.reduced = True
        return ncat

    def reduce_catalog(self, other, reduce_column, wout_QF=False, masked=False):
        
        red_val = column_dict[reduce_column]
        self_numbers = np.array([])
        other_numbers = np.array([])

        for i, gal in enumerate(self.raw_catalog[0]):
            if not (self.reduced):
                if(wout_QF):
                    if (self.raw_catalog[column_dict['QF']][i].astype(float) > 0):
                        continue
                
            if gal in other.raw_catalog[0]:
                index_self = np.where(self.raw_catalog[0] == gal)
                index_other = np.where(other.raw_catalog[0] == gal)
                self_numbers = np.append(self_numbers, index_self)
                other_numbers = np.append(other_numbers, index_other)
                continue                
        
        new_catalog = []
        
        self_galaxies = [self.raw_catalog[0][j] for j in self_numbers]
        
        new_catalog.append(self_galaxies)
        if(self.reduced):
            shape = self.raw_catalog.shape[0]
            for i in range(1, shape):
                self_column = [float(self.raw_catalog[i][j]) for j in self_numbers]
                new_catalog.append(self_column)
        else:
            self_column = [float(self.raw_catalog[red_val][j]) for j in self_numbers]
            new_catalog.append(self_column)
        
        if(other.reduced):
            shape = other.raw_catalog.shape[0]-1
            for i in range(1, shape):
                other_column = [float(other.raw_catalog[i][j]) for j in other_numbers]
                new_catalog.append(other_column)
        else:
            other_column = [float(other.raw_catalog[red_val][j]) for j in other_numbers]
            new_catalog.append(other_column)
        
        
        new_catalog = catalog(np.array(new_catalog))
        new_catalog.reduced = True
        return new_catalog
        
    
    def load_catalog(self, path):
        cat = np.loadtxt(path, delimiter=',', dtype='str').T
        self.raw_catalog = cat
    
    def __init__(self, external_catalog=None):
        self.raw_catalog = external_catalog
        self.reduced = False
        
        
        
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