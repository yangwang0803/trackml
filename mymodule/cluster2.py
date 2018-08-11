
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
        
        # rotate_on_r parameters
        self.use_rotate_on_r = parameter['use_rotate_on_r']
        self.rotate_on_r_w1 = parameter['rotate_on_r_w1']
        self.rotate_on_r_w2 = parameter['rotate_on_r_w2']
        self.rotate_on_r_w3 = parameter['rotate_on_r_w3']
        self.rotate_on_r_quad_coef = parameter['rotate_on_r_quad_coef']
        self.rotate_on_r_max_iter = parameter['rotate_on_r_max_iter']
        self.rotate_on_r_eps = parameter['rotate_on_r_eps']
        self.rotate_on_r_step_eps = parameter['rotate_on_r_step_eps']
        
        # rotate_on_theta parameters
        self.use_rotate_on_theta = parameter['use_rotate_on_theta']
        self.rotate_on_theta_eps = parameter['rotate_on_theta_eps']
        self.rotate_on_theta_max_iter = parameter['rotate_on_theta_max_iter']
        
        self.use_shift_on_z = parameter['use_shift_on_z']
        
        self.use_multiprocess = parameter['use_multiprocess']
        
    def __run_DBSCAN(self, dbscan_input):
        cl = DBSCAN(eps=dbscan_input[0], min_samples=1, metric='euclidean', algorithm='auto', n_jobs=4)
        l = cl.fit_predict(dbscan_input[1])
        
        return l

    def run_rotation(self, hits, z_0):
        results = []

        if self.use_rotate_on_r:
            results += self.__rotate_on_r(hits, z_0)
            
        if self.use_rotate_on_theta:
            results += self.__rotate_on_theta(hits, z_0)
            
        return results
    
    
    def run(self):
        
        results = []
        
        if self.use_shift_on_z==False:
            results += self.run_rotation(self.hits, 0)
        else:
            z_0_list = [0, 2, -2, 5,-5, 10, -10]
            
            if self.use_multiprocess:
                with ProcessPoolExecutor(max_workers=7) as executor:
                    future_to_rs = {executor.submit(self.run_rotation, self.hits, z_0): z_0 for z_0 in z_0_list}
                    for future in as_completed(future_to_rs):
                        rs = future_to_rs[future]
                        try:
                            results += future.result()
                        except Exception as exc:
                            print('exception: %s', exc)
            else:
                for z_0 in z_0_list:
                    results += self.run_rotation(self.hits, z_0)    
        
        return results
    
    # Rotate based on theta_0
    def __rotate_on_theta(self, hits, z_0):
        
        results = []
        dbscan_inputs = []
        
        x  = hits.x.values
        y  = hits.y.values
        z  = hits.z.values + z_0
        r  = np.sqrt(x**2+y**2)
        theta  = np.arctan2(y,x)
        
        for theta_0 in np.linspace(start=-np.pi, stop=np.pi, num=self.rotate_on_theta_max_iter, endpoint=False):
            
            cos = np.cos(theta - theta_0)
            index=np.where(cos>0.0001)[0]
            
            r_0 = r[index]/cos[index]/2
            x_0 = r_0*np.cos(theta_0)
            y_0 = r_0*np.sin(theta_0)
            alpha = np.arctan2(y[index]-y_0, x[index]-x_0)
            alpha_0 = np.arctan2(-y_0, -x_0)
            
            angle = alpha-alpha_0
            angle[angle>np.pi] -= 2*np.pi
            angle[angle<-np.pi] += 2*np.pi

            f1 = 1/r_0
            f2 = np.arctan2(np.absolute(angle)*r_0, z[index])
            f3 = np.sign(angle)
             
            X = StandardScaler().fit_transform(np.column_stack([
                f1, f2, f3
            ]))

            dbscan_inputs.append((self.rotate_on_theta_eps, X, index))     

        # DBSCAN
        for dbscan_input in tqdm(dbscan_inputs):
            l2 = self.__run_DBSCAN(dbscan_input)
            
            l = np.zeros((len(hits,)))
            index = dbscan_input[2]
            l[index] = l2
            results.append(l)

        return results
    
    # Rotate based on r
    def __rotate_on_r(self, hits, z_0):
        
        results = []
        dbscan_inputs = []
        
        x  = hits.x.values
        y  = hits.y.values
        z  = hits.z.values + z_0
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