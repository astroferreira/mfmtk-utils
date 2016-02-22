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
    
    def reduce_catalog(self, other, reduce_column, wout_QF=False):
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