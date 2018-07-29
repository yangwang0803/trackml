
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
from operator import itemgetter
from multiprocessing import Pool

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm


# In[ ]:


class Cluster():
    
    def __init__(self, hits, parameter):
        
        self.hits = hits
        
        self.use_rotate_on_z = parameter['use_rotate_on_z']
        self.rotate_on_z_w1 = parameter['rotate_on_z_w1']
        self.rotate_on_z_w2 = parameter['rotate_on_z_w2']
        self.rotate_on_z_w3 = parameter['rotate_on_z_w3']
        self.rotate_on_z_max_iter = parameter['rotate_on_z_max_iter']
        self.rotate_on_z_dz0 = parameter['rotate_on_z_dz0']
        self.rotate_on_z_step_dz = parameter['rotate_on_z_step_dz']
        self.rotate_on_z_eps = parameter['rotate_on_z_eps']
        self.rotate_on_z_step_eps = parameter['rotate_on_z_step_eps']
        
        self.use_rotate_on_r = parameter['use_rotate_on_r']
        self.rotate_on_r_w1 = parameter['rotate_on_r_w1']
        self.rotate_on_r_w2 = parameter['rotate_on_r_w2']
        self.rotate_on_r_w3 = parameter['rotate_on_r_w3']
        self.rotate_on_r_quad_coef = parameter['rotate_on_r_quad_coef']
        self.rotate_on_r_max_iter = parameter['rotate_on_r_max_iter']
        self.rotate_on_r_eps = parameter['rotate_on_r_eps']
        self.rotate_on_r_step_eps = parameter['rotate_on_r_step_eps']
        
        self.use_shift_on_z = parameter['use_shift_on_z']
        
        self.use_multiprocess = parameter['use_multiprocess']
        
    def __run_DBSCAN(self, dbscan_input):
        cl = DBSCAN(eps=dbscan_input[0], min_samples=1, metric='euclidean', algorithm='auto', n_jobs=4)
        l = cl.fit_predict(dbscan_input[1])
        return l
        
    # Rotate based on z
    def __rotate_on_z(self, hits):
               
        results = []
        dbscan_inputs = []
        
        x  = hits.x.values
        y  = hits.y.values
        z  = hits.z.values
        r  = np.sqrt(x**2+y**2)
        d  = np.sqrt(x**2+y**2+z**2)
        a0  = np.arctan2(y,x)
        z1 = z/r
        z2 = z/d
        direction = 1

        for i in range(self.rotate_on_z_max_iter):
            direction = (-1)*direction
            dz = direction*(self.rotate_on_z_dz0+i*self.rotate_on_z_step_dz)
            a1 = a0 + np.sign(z)*dz*z

            f1 = np.sin(a1)
            f2 = np.cos(a1)
            f3 = z1
            f4 = z2
            f5 = r/d

            X = StandardScaler().fit_transform(np.column_stack([
                f1, f2, f3, f4, f5
            ]))

            # Manual Scale
            scale = [1.0,    # f1 
                     1.0,    # f2
                     self.rotate_on_z_w1,    # f3
                     self.rotate_on_z_w2,    # f4
                     self.rotate_on_z_w3]    # f5

            for j in range(0, X.shape[1]):
                X[:,j] *=scale[j]

            dbscan_inputs.append((self.rotate_on_z_eps+i*self.rotate_on_z_step_eps, X))

        # DBSCAN
        for dbscan_input in tqdm(dbscan_inputs):
            l = self.__run_DBSCAN(dbscan_input)
            results.append(l)
                
        return results
    
    # Rotate based on r
    def __rotate_on_r(self, hits):
        
        results = []
        dbscan_inputs = []
        
        x  = hits.x.values
        y  = hits.y.values
        z  = hits.z.values
        r  = np.sqrt(x**2+y**2)
        d  = np.sqrt(x**2+y**2+z**2)
        a0  = np.arctan2(y,x)
        z1 = z/r
        z2 = z/d
        direction = 1
        
        for i in range(self.rotate_on_r_max_iter):
            direction = (-1)*direction
            a1=a0+direction*(r+self.rotate_on_r_quad_coef*r**2)/1000*(i/2)/180*np.pi
            
            f1 = np.sin(a1)
            f2 = np.cos(a1)
            f3 = z1
            f4 = z2
            f5 = r/d
             
            X = StandardScaler().fit_transform(np.column_stack([
                f1, f2, f3, f4, f5
            ]))
            
            # Manual Scale
            scale = [1.0,    # f1 
                     1.0,    # f2
                     self.rotate_on_r_w1,    # f3
                     self.rotate_on_r_w2,    # f4
                     self.rotate_on_r_w3]    # f8
            
            for j in range(0, X.shape[1]):
                X[:,j] *=scale[j]

            dbscan_inputs.append((self.rotate_on_r_eps+i*self.rotate_on_r_step_eps, X))

        # DBSCAN
        for dbscan_input in tqdm(dbscan_inputs):
            l = self.__run_DBSCAN(dbscan_input)
            results.append(l)

        return results
    
    
    def run_rotation(self, hits):
        results = []
        
        if self.use_rotate_on_z:
            results += self.__rotate_on_z(hits)

        if self.use_rotate_on_r:
            results += self.__rotate_on_r(hits)
            
        return results
    
    
    def run(self):
             
        results = []
        
        if self.use_shift_on_z==False:
            results += self.run_rotation(self.hits)
        else:
            dz_list = [0, 2, -2, 5,-5, 10,-10]
            hits_list = [self.hits.assign(z=self.hits.z+dz) for dz in dz_list]
            
            if self.use_multiprocess:
                with ProcessPoolExecutor(max_workers=7) as executor:
                    future_to_rs = {executor.submit(self.run_rotation, hits): hits for hits in hits_list}
                    for future in as_completed(future_to_rs):
                        rs = future_to_rs[future]
                        try:
                            results += future.result()
                        except Exception as exc:
                            print('exception: %s', exc)
            else:
                for hits in hits_list:
                    results += self.run_rotation(hits)
        
        return results

