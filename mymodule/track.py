import pandas as pd
import numpy as np
from scipy import optimize
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.linear_model import LinearRegression


class Track():
    
    def __init__(self, data, parameter):
        
        # unpack parameter
        self.is_training = parameter['is_training']
        
        # track attribute
        self.list_hits = frozenset(data.hit_id.values)
        assert len(self.list_hits)==len(data)
        self.track_id = hash(self.list_hits)
        self.length_1 = len(data)
        self.length_2 = len(data.groupby(['volume_id', 'layer_id', 'module_id']))
        self.length_diff = self.length_1-self.length_2
        
        # private copy of the data
        self.__data = data
        self.__data_is_called = False
        self.__update_is_called = False
    
    @property
    def data(self):
        
        # only enrich data when data is called the first time
        if self.__data_is_called == False:
            if self.__data.z.mean()>0:
                self.__data.sort_values(by='z', inplace=True)
            else:
                self.__data.sort_values(by='z', ascending=False, inplace=True)

            self.__data = self.__data.assign(track_id = self.track_id)
            self.__data = self.__data.assign(length_1 = self.length_1)
            self.__data = self.__data.assign(length_2 = self.length_2)
            residual_1, residual_2, residual_3 = self.__residual()
            self.__data = self.__data.assign(residual_1 = residual_1)
            self.__data = self.__data.assign(residual_2 = residual_2)
            self.__data = self.__data.assign(residual_3 = residual_3)
            self.__data = self.__data.assign(outlier = self.__detect_outlier())
            self.__data = self.__data.assign(collision = self.__detect_collision())
            self.__data_is_called == True
        
        return self.__data
    
    def update(self):
        
        if self.__data_is_called == False:
            self.data
            
        self.__data = self.__data.loc[(self.__data.outlier==False) & (self.__data.collision==False)]
        
        self.list_hits = frozenset(self.__data.hit_id.values)
        assert len(self.list_hits)==len(self.__data)
        self.track_id = hash(self.list_hits)
        self.length_1 = len(self.__data)
        self.length_2 = len(self.__data.groupby(['volume_id', 'layer_id', 'module_id']))
        self.length_diff = self.length_1-self.length_2
        
        # different hits may have exactly the same x,y,z and cell information
        # assert self.length_diff == 0
        
        self.__data = self.__data.assign(track_id = self.track_id)
        self.__data = self.__data.assign(length_1 = self.length_1)
        self.__data = self.__data.assign(length_2 = self.length_2)
        
        self.__update_is_called = True
        
        return self.__data
    
    def __detect_outlier(self):
        
        w = np.linspace(start=1, stop=1.5, num=len(self.__data), endpoint=True)
        r_1_1 = 10
        r_1_2 = 5
        r_1_3 = 6
        
        r_2_1 = 5
        r_2_2 = 3
        r_2_3 = 4
        
        rule_1 = (self.__data.residual_1 > r_1_1*w*self.__data.residual_1.median()) 
        rule_1 = rule_1 | (self.__data.residual_2 > r_1_2*w*self.__data.residual_2.median())
        rule_1 = rule_1 | (self.__data.residual_3 > r_1_3*w*self.__data.residual_3.median())
        
        rule_2 = (self.__data.residual_1 > r_2_1*w*self.__data.residual_1.median()) 
        rule_2 = rule_2 & (self.__data.residual_2 > r_2_2*w*self.__data.residual_2.median())
        rule_2 = rule_2 & (self.__data.residual_3 > r_2_3*w*self.__data.residual_3.median())
        
        return rule_1 | rule_2
    
    def __detect_collision(self):
               
        temp = self.__data[['volume_id', 'layer_id', 'module_id', 'residual_1', 'residual_2']]
        temp = temp.assign(r_1 = temp.residual_1.transform(lambda x:x/x.median()))
        temp = temp.assign(r_2 = temp.residual_2.transform(lambda x:x/x.median()))

        if temp.r_1.mean()>temp.r_2.mean():
            return temp.groupby(['volume_id', 'layer_id', 'module_id']).r_1.transform(lambda x: x>x.min())
        else:
            return temp.groupby(['volume_id', 'layer_id', 'module_id']).r_2.transform(lambda x: x>x.min())      
    
    def __residual(self):
        
        if len(self.__data) < 6:
            return 9999,9999,9999
        
        # fit x-y circle
        x = self.__data.x.values
        y = self.__data.y.values

        x_m = np.mean(x)
        y_m = np.mean(y)

        def calc_R(c):
            """ calculate the distance of each 2D point from the center c=(xc, yc) """
            return np.sqrt((x-c[0])**2 + (y-c[1])**2)

        def calc_ecart(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(calc_ecart, center_estimate)

        xc, yc = center
        Ri = calc_R(center)
        R = Ri.mean()
        residual_1 = np.absolute(Ri - R)

        self.fit_xy = (xc,yc,R)
        
        # fit x-z polynomial
        z = self.__data.z.values
        zz = z**2
        zz = zz-zz.mean()
        w = np.linspace(start=1, stop=1.5, num=len(x), endpoint=True)
        w = 1/w
        w = w/sum(w)

        X = np.column_stack((z,zz))
        model = LinearRegression().fit(X,x,sample_weight=w)
        x_pred = model.predict(X)
        residual_3 = np.absolute(x_pred-x)
        
        self.fit_zx = x_pred
        
        # fit a-z line
        x = x-xc
        y = y-yc
        a = np.arctan2(x,y).reshape(-1, 1)
        b = np.sign(a).reshape(-1,1)

        X = np.column_stack((a,b))
        model = LinearRegression().fit(X,z,sample_weight=w)
        z_pred = model.predict(X)
        residual_2 = np.absolute(z_pred-z)
        
        self.fit_az = z_pred
        
        return residual_1, residual_2, residual_3
    
    def plot(self):
        
        if self.__update_is_called == False:
            x = self.__data.x.values
            y = self.__data.y.values
            a = np.arctan2(x-self.fit_xy[0],y-self.fit_xy[1])
            z = self.__data.z.values

            if self.is_training:
                color = [(particle_id%97) for particle_id in self.__data.particle_id.values]

            plt.figure(figsize=(16,8))

            # 1st plot
            ax = plt.subplot(2, 3, 1)

            if self.is_training:
                ax.scatter(x, y, c=color)
            else:
                ax.scatter(x, y)

            ax.set_xlabel('x')
            ax.set_ylabel('y')

            # 2nd plot
            ax = plt.subplot(2, 3, 2)

            ax.plot(x, y, '-o')

            ax.set_xlabel('x')
            ax.set_ylabel('y')

            # 3rd plot
            ax = plt.subplot(2, 3, 3)

            if self.is_training:
                ax.scatter(z, x, c=color)
                ax.plot(z, self.fit_zx, '-')
            else:
                ax.scatter(z, x)
                ax.plot(z, self.fit_zx, '-')

            ax.set_xlabel('z')
            ax.set_ylabel('x')

            # 4nd plot
            ax = plt.subplot(2, 3, 4)

            if self.is_training:
                ax.scatter(z, y, c=color)
            else:
                ax.scatter(z, y)

            ax.set_xlabel('z')
            ax.set_ylabel('y')

            # 5th plot
            ax = plt.subplot(2, 3, 5)

            if self.is_training:
                ax.scatter(a, z, c=color)
                ax.plot(a, self.fit_az, '-')
            else:
                ax.scatter(a, z)
                ax.plot(a, self.fit_az, '-')

            ax.set_xlabel('a')
            ax.set_ylabel('z')

            # 6th plot
            ax = plt.subplot(2, 3, 6, projection='3d')

            if self.is_training:
                ax.scatter(z, x, y, '-o', c=color, alpha=0.5)
            else:
                ax.scatter(z, x, y, '-o', alpha=0.5)

            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')

            plt.show()            

class TrackPool():
    
    def __init__(self, data, results, parameter):
        
        self.track_length_1_min = parameter['track_length_1_min']
        self.use_multiprocess = parameter['use_multiprocess']
        self.is_training = parameter['is_training']
        
        self.length_results = len(results)
        self.list_tracks = self.__createPool(data, results, parameter)
                                           
        
    def __createPool(self, data, results, parameter):
        
        pool = []
        
        data.set_index('hit_id', inplace=True)
        
        if self.use_multiprocess:
            with ProcessPoolExecutor(max_workers=7) as executor:
                future_to_rs = {executor.submit(self.extractTracks, data, result, parameter): result for result in results}
                for future in as_completed(future_to_rs):
                    rs = future_to_rs[future]
                    try:
                        pool += future.result()
                    except Exception as exc:
                        print('exception: %s', exc)
        else:
            for i in tqdm(range(0, len(results))):
                pool += self.extractTracks(data, results[i], parameter)

        self.length_1 = len(pool)
        pool = self.__removeDuplicates(pool)
        self.length_2 = len(pool)
        
        data.reset_index(inplace=True)
        
        return pool
    
    def extractTracks(self, data, result, parameter):
        
        tracks = []
        
        temp = pd.DataFrame(result, columns=['label'])
        temp = temp.assign(hit_id=data.index.values)
        ranking = temp.groupby('label').hit_id.count().reset_index()

        list_label = ranking.loc[(ranking.hit_id >= self.track_length_1_min) & (ranking.hit_id <= 20)].label.values

        for label in list_label:
            
            df = temp.loc[temp.label==label].set_index('hit_id').join(data, how='inner')

            parameter = {}
            parameter['is_training'] = self.is_training

            tracks.append(Track(df.reset_index(), parameter))
        
        return tracks
    
    def __removeDuplicates(self, pool):
        
        n = len(pool)
        pool.sort(key=lambda x: x.track_id)
        
        last = pool[0]
        lasti = i = 1
        
        while i < n:
            if pool[i].track_id != last.track_id:
                pool[lasti] = last = pool[i]
                lasti += 1
            i += 1
        return pool[:lasti]
                