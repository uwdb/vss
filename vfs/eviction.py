import logging

EVICTION_POLICY_LRU = 'LRU'
EVICTION_POLICY_VSS = 'VSS_LRU'

def get_gop_eviction_candidate(database, eviction_policy, temporal_eviction_weight, quality_eviction_weight):
    logical_id, consumed = get_overbudget_logical_video(database)

    if logical_id is not None:
        return get_violating_gop(database, logical_id, eviction_policy, temporal_eviction_weight, quality_eviction_weight)
    else:
        return None, None

def evict_gop(database, gop_id, cut):
    from vfs.gop import Gop
    from vfs.physicalvideo import PhysicalVideo

    evict_gop = Gop.get(gop_id)
    gop_count = database.execute("SELECT COUNT(*) FROM gops WHERE physical_id = ?", evict_gop.physical_id).fetchone()[0]

    if gop_count == 1:
        PhysicalVideo.delete(evict_gop.video())
        #logging.info('EvictWorker: Evicting GOP %d (covered by %d; last GOP, also deleting physical video)', gop_id, cover_gop_id)
        logging.info('EvictWorker: Evicting GOP %d (last GOP, also deleting physical video)', gop_id)
    elif cut:
        new_physical_video = PhysicalVideo.add(
            evict_gop.video().logical(),
            evict_gop.video().height,
            evict_gop.video().width,
            evict_gop.video().codec)
        database.executebatch([
            'UPDATE gops SET physical_id = {} WHERE physical_id = {} AND start_time < {}'.format(
                new_physical_video.id, evict_gop.physical_id, evict_gop.start_time),
            'DELETE FROM gops WHERE id = {}'.format(gop_id)])
        Gop.delete(evict_gop)
        #logging.info('EvictWorker: Evicting GOP %d (covered by %d; middle GOP, cutting physical video)', gop_id, cover_gop_id)
        logging.info('EvictWorker: Evicting GOP %d (middle GOP, cutting physical video)', gop_id)
    else:
        #logging.info('EvictWorker: Evicting GOP %d (covered by %d; endpoint GOP)', gop_id, cover_gop_id)
        logging.info('EvictWorker: Evicting GOP %d (endpoint GOP)', gop_id)
        Gop.delete(evict_gop)

def get_overbudget_logical_video(database):
    return (database.execute(
        'SELECT logical_videos.id, SUM(size) / (budget+1.0) AS consumed '
        'FROM logical_videos, physical_videos, gops '
        'WHERE logical_videos.id = physical_videos.logical_id AND '
        '      gops.physical_id = physical_videos.id '
        'GROUP BY logical_videos.id '
        #'WHERE consumed > 1 '
        'ORDER BY consumed DESC '
        'LIMIT 1'
    ).fetchone() or [None, None])

def get_violating_gop(database, logical_id, policy, temporal_eviction_weight, quality_eviction_weight):
    if policy == EVICTION_POLICY_VSS:
        return get_lru_vss_gop(database, logical_id, temporal_eviction_weight, quality_eviction_weight)
    elif policy == EVICTION_POLICY_LRU:
        return get_lru_gop(database, logical_id)
    else:
        raise Exception(f'Unsupported eviction policy {policy}')

def get_lru_vss_gop(database, logical_id, temporal_eviction_weight, quality_eviction_weight):
    gop_id, cut, time_percentile, quality_percentile = (database.execute(
        'SELECT gop_ranks.id, '
        '       gop_ranks.requires_cut, '
        '       (start_time - min_physical_time) / physical_duration AS time_percentile, '
        '       CAST(lower_quality_count AS float) / (SELECT COUNT(*) FROM gops all_gops WHERE all_gops.physical_id = gop_ranks.physical_id) AS quality_percentile '
        'FROM gop_ranks '
        'INNER JOIN physical_video_times physical_videos ON gop_ranks.physical_id = physical_videos.id '
        'WHERE gop_ranks.logical_id = ? '
        'ORDER BY clock + ? * (1 - ABS(0.5 - time_percentile)) + ? * (1 - quality_percentile) ASC '
        'LIMIT 1',
        (logical_id,
         temporal_eviction_weight,
         quality_eviction_weight)).fetchone() or [None, None, None, None])

    return gop_id, bool(cut)

def get_lru_gop(database, logical_id):
    gop_id, physical_id, cut, _, _, cover_gop_id, gop_count = (database.execute(
        'SELECT gop_ranks.id,'
        '       gop_ranks.physical_id, '
        '       start_time != min_physical_time AND end_time != max_physical_time AS requires_cut, '
        '       cover_gop_id, '
        '       (SELECT COUNT(*) FROM gops gopcount WHERE gopcount.physical_id = gop_ranks.physical_id) AS gop_count '
        'FROM gop_ranks '
        'INNER JOIN physical_video_times '
        '  ON gop_ranks.physical_id = physical_video_times.id '
        'WHERE physical_video_times.logical_id = ? AND '
        '      cover_gop_id IS NOT NULL '
        'ORDER BY clock ASC '
        'LIMIT 1',
        logical_id).fetchone() or [None, None, None, None, None, None, None])

    return gop_id, (physical_id, cut, cover_gop_id, gop_count)

def get_lru_vss_gop_old(database, logical_id, temporal_eviction_weight, quality_eviction_weight):
    gop_id, physical_id, cut, _, _, cover_gop_id, gop_count = (database.execute(
        'SELECT gops.id, '
        '       gop_ranks.physical_id, '
        '       start_time != min_physical_time AND end_time != max_physical_time AS requires_cut, '
        '       midpoint_time / physical_duration AS time_percentile, '
        '       quality_percentile, '
        '        ((end_time - start_time) / 2.0) / physical_duration AS time_percentile, '
        '        CAST(lower_quality_count AS float) / (SELECT COUNT(*) FROM gops all_gops WHERE all_gops.physical_id = gops.physical_id) AS quality_percentile '
        '       cover_gop_id, '
        '       (SELECT COUNT(*) FROM gops gopcount WHERE gopcount.physical_id = gop_ranks.physical_id) AS gop_count '
        'FROM gop_ranks '
        'INNER JOIN gops oN gop_ranks.id = gops.id '
        'INNER JOIN physical_video_times physical_videos ON gops.physical_id = physical_videos.id '
        'INNER JOIN physical_video_times '
        '  ON gop_ranks.physical_id = physical_video_times.id '
        'WHERE gop_ranks.logical_id = ? AND '
        '      cover_gop_id IS NOT NULL '
        'ORDER BY clock + ? * (1 - ABS(0.5 - time_percentile)) + ? * (1 - quality_percentile) ASC '
        'LIMIT 1',
        (logical_id,
         temporal_eviction_weight,
         quality_eviction_weight)).fetchone() or [None, None, None, None])

    return gop_id, (physical_id, cut, cover_gop_id, gop_count)
