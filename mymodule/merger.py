import pandas as pd

from mymodule.track import TrackPool

# Merge long/good track
def Merge_1(pool, parameter):
    
    # Unpack parameter
    track_length_2_min = parameter['track_length_2_min']
    track_length_2_max = parameter['track_length_2_max']
    track_length_diff_max = parameter['track_length_diff_max']
    
    tracks = pool.list_tracks
    tracks = list(track for track in tracks if track.length_diff<=track_length_diff_max)
    tracks = list(
        track for track in tracks \
        if track.length_2 >= track_length_2_min and track.length_2 <= track_length_2_max)
    
    df = pd.concat(list(track.update() for track in tracks))
    df = df.drop_duplicates(subset=['hit_id', 'track_id'])
    
    assert (df.outlier==True).any() | (df.collision==True).any() == False    
    
    # If a hit_id has multiple track_id, pick the one with longest length_2
    df = df.sort_values(by=['hit_id', 'length_2'], ascending=False)
    output = df.groupby('hit_id').track_id.first().reset_index()
    
    # Remove track with track length < track_length_2_min
    output = output.assign(length = output.groupby('track_id').transform(lambda x:len(x)))
    output = output.loc[output.length >= track_length_2_min - track_length_diff_max]
    output = output.drop(columns='length')
    
    return output


def Merge_2(data):
    
    output = data.loc[data.track_id.isnull()]
    output = output.assign(track_id=output.apply(lambda row: hash(frozenset([int(row.hit_id)])), axis=1))
    
    return output[['hit_id', 'track_id']]