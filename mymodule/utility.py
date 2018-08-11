import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree

from trackml.score import score_event


def CreateSubmission(event_id, data):
    
    temp = np.column_stack(([event_id]*len(data), data.hit_id.values, data.track_id.fillna(0).values))
    submission = pd.DataFrame(data=temp, columns=["event_id", "hit_id", "track_id"]).astype(int)
    
    return submission

def Score(event_id, data, truth):
    
    submission = CreateSubmission(event_id, data)
    
    return score_event(truth, submission)

def FixTrackID(submission):
    
    unique, inverse = np.unique(submission.track_id, return_inverse=True)
    
    return submission.assign(track_id=inverse)
    
def Extend(data, limit=0.04, num_neighbours=18):
    
    submission = CreateSubmission(0, data)
    hits = data[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id']]
    
    df = submission.merge(hits,  on=['hit_id'], how='left')
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))

    for angle in range(-90,90,1):

        #df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)]
        df1 = df.loc[(df.arctan2>(angle-1.5)/180*np.pi) & (df.arctan2<(angle+1.5)/180*np.pi)]

        min_num_neighbours = len(df1)
        if min_num_neighbours<3: continue

        hit_ids = df1.hit_id.values
        x,y,z = df1[['x', 'y', 'z']].values.T
        r  = (x**2 + y**2)**0.5
        r  = r/1000
        a  = np.arctan2(y,x)
        c = np.cos(a)
        s = np.sin(a)
        tree = KDTree(np.column_stack([c, s, r]), metric='euclidean')


        track_ids = list(df1.track_id.unique())
        num_track_ids = len(track_ids)
        min_length=3

        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue

            idx = np.where(df1.track_id==p)[0]
            if len(idx)<min_length: continue

            if angle>0:
                idx = idx[np.argsort( z[idx])]
            else:
                idx = idx[np.argsort(-z[idx])]

            ## start and end points  ##
            idx0,idx1 = idx[0],idx[-1]
            a0 = a[idx0]
            a1 = a[idx1]
            r0 = r[idx0]
            r1 = r[idx1]
            c0 = c[idx0]
            c1 = c[idx1]
            s0 = s[idx0]
            s1 = s[idx1]

            da0 = a[idx[1]] - a[idx[0]]  #direction
            dr0 = r[idx[1]] - r[idx[0]]
            direction0 = np.arctan2(dr0,da0)

            da1 = a[idx[-1]] - a[idx[-2]]
            dr1 = r[idx[-1]] - r[idx[-2]]
            direction1 = np.arctan2(dr1,da1)

            ## extend start point
            ns = tree.query([[c0, s0, r0]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
            ns = np.concatenate(ns)

            direction = np.arctan2(r0 - r[ns], a0 - a[ns])
            diff = 1 - np.cos(direction - direction0)
            ns = ns[(r0 - r[ns] > 0.01) & (diff < (1 - np.cos(limit)))]
            for n in ns: df.loc[df.hit_id == hit_ids[n], 'track_id'] = p

            ## extend end point
            ns = tree.query([[c1, s1, r1]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
            ns = np.concatenate(ns)

            direction = np.arctan2(r[ns] - r1, a[ns] - a1)
            diff = 1 - np.cos(direction - direction1)
            ns = ns[(r[ns] - r1 > 0.01) & (diff < (1 - np.cos(limit)))]
            for n in ns:  df.loc[df.hit_id == hit_ids[n], 'track_id'] = p

    df = df[['hit_id', 'track_id']]
    temp = data.merge(df, how='left', on='hit_id', suffixes=('_old', ''))
    data.update(temp.track_id)
    
    return data
    

