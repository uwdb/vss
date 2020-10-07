import time
import os
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from vfs.db import Database

DEFAULT_PERFECT_QUALITY_THRESHOLD = 100
DEFAULT_TEMPORAL_EVICTION_WEIGHT = 4
DEFAULT_QUALITY_EVICTION_WEIGHT = 2

class VFS(object):
    class __Singleton(object):
        DEFAULT_PATH = 'data'
        DATABASE_NAME = 'vfs.db'

        def __init__(self, transient=False,
                     path=DEFAULT_PATH, database_filename=os.path.join(DEFAULT_PATH, DATABASE_NAME),
                     temporal_eviction_weight=DEFAULT_TEMPORAL_EVICTION_WEIGHT,
                     quality_eviction_weight=DEFAULT_QUALITY_EVICTION_WEIGHT,
                     perfect_quality_threshold=DEFAULT_PERFECT_QUALITY_THRESHOLD):
            from vfs.jointcompression import JointCompression

            logging.debug('VFS: data path "%s"', path)
            logging.debug('VFS: database "%s"', database_filename)

            self.path = path
            self.perfect_quality_threshold = perfect_quality_threshold
            self.temporal_eviction_weight = temporal_eviction_weight
            self.quality_eviction_weight = quality_eviction_weight
            self.clock = 0
            self.engine_lock = threading.Lock()
            self.database = Database(self, database_filename)
            self.compression = JointCompression()

            if not os.path.exists(self.path):
                os.mkdir(self.path)

            self.running = True
            self.pid_filename = os.path.join(self.path, 'pid')
            self.transient = transient or self.is_engine_running()
            if not self.transient:
                logging.info('VFS: starting engine')

                with open(self.pid_filename, 'w') as f:
                    f.write(str(os.getpid()))

                self._metadata_worker = threading.Thread(target=self._create_metadata, args=(self,))
                self._compression_worker = threading.Thread(target=self._apply_compression, args=(self,))
                self._budget_worker = threading.Thread(target=self._apply_budget, args=(self,))
                self._compaction_worker = threading.Thread(target=self._apply_compaction, args=(self,))
                self._error_worker = threading.Thread(target=self._apply_compute_error, args=(self,))
                self._eviction_worker = threading.Thread(target=self._apply_eviction, args=(self,))

                if not self.transient:
                    self._metadata_worker.start()
                    self._compression_worker.start()
                    self._budget_worker.start()
                    self._compaction_worker.start()
                    self._error_worker.start()
                    self._eviction_worker.start()
            else:
                logging.debug('VFS: engine already running')

        def close(self):
            self.running = False
            if not self.transient:
                logging.info('VFS: stopping engine')
                self._metadata_worker.join()
                self._compression_worker.join()
                self._budget_worker.join()
                self._compaction_worker.join()
                self._error_worker.join()
                self._eviction_worker.join()
                os.remove(self.pid_filename)
                logging.debug('VFS: engine terminated')

        def is_engine_running(self):
            if not os.path.exists(self.pid_filename):
                return False
            with open(self.pid_filename, 'r') as f:
                pid = int(f.read())
            return psutil.pid_exists(pid)

        @classmethod
        def _sleep(cls, instance, n):
            for _ in range(int(n) * 2):
                time.sleep(0.5)
                if not instance.running:
                    break

        @classmethod
        def _create_metadata(cls, instance, poll_interval=0.001, maximum_interval=16, pool_size=10):
            from vfs.physicalvideo import PhysicalVideo
            from vfs.histogram import Histogram
            from vfs.descriptor import Descriptor

            last_physical_id, last_gop_id = -1, -1
            current_interval = poll_interval
            futures = {}

            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                while instance.running:
                    if len(futures) < pool_size:
                        physical_id, gop_id, filename, height, width, codec = (instance.database.execute(
                            'SELECT physical_id, gops.id, filename, height, width, codec FROM gops '
                            'INNER JOIN physical_videos '
                            '  ON gops.physical_id = physical_videos.id '
                            'WHERE examined = 0 AND joint = 0 AND histogram IS NULL AND '
                            '      physical_id != ? AND ' #AND gops.id > ? '
                            '      NOT gops.id IN ({}) '
                            'ORDER BY gops.id '
                            'LIMIT 1'.format(','.join(map(str, futures.keys()))), last_physical_id).fetchone() \
                                                                               or [-1, None, None, None, None, None])

                        if gop_id is not None:
                            logging.debug('MetadataWorker: creating metadata for GOP %d.%d', physical_id, gop_id)
                            #last_gop_id = gop_id
                            last_physical_id = physical_id

                            def create_metadata(gop_id, height, width, codec):
                                histogram, frame = Histogram.create(filename, (height, width, PhysicalVideo.CHANNELS), codec)
                                keypoints, descriptors = Descriptor.create(frame)
                                instance.database.execute(
                                    'UPDATE gops SET histogram = ?, keypoints = ?, descriptors = ? WHERE id = ?',
                                    (histogram, keypoints, descriptors, gop_id))
                            futures[gop_id] = pool.submit(create_metadata, gop_id, height, width, codec)
                            current_interval = poll_interval
                        else:
                            #logging.debug('MetadataWorker: no gops found')
                            current_interval = min(current_interval * 2, maximum_interval)
                            last_physical_id = -1

                    for gop_id, future in [(gid, f) for gid, f in futures.items() if f.done()]:
                        del futures[gop_id]
                        #futures.remove(future)

                    cls._sleep(instance, current_interval)

        @classmethod
        def _apply_compression(cls, instance, poll_interval=1, maximum_interval=8, pool_size=8, cluster_period=4):
            from vfs.histogram import Histogram
            from vfs.jointcompression import JointCompression
            from vfs.physicalvideo import PhysicalVideo

            current_interval = poll_interval
            iteration = 0
            clusters = 0
            epoch = 0
            total_bytes_saved = 0

            #cluster_period = 1
            futures = []

            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                while instance.running:
                    if iteration % cluster_period == 0 and PhysicalVideo.count() > 0: # and instance.database.execute(
                            #'SELECT 1 FROM gops WHERE histogram IS NULL LIMIT 1').fetchone() is None:s                      logging.info("CompressionWorker: reclustering")
                        clusters = Histogram.cluster_all()
                        if instance.database.execute('SELECT 1 FROM gops WHERE examined <= ? LIMIT 1', epoch)\
                                .fetchone() is None:
                            logging.info("CompressionWorker: ending epoch %d", epoch)
                            epoch += 1

                    if clusters > 0 and len(futures) < 5:
                            #pool.submit(JointCompression.execute, epoch)
                            success, target, future = JointCompression.execute(epoch, pool)
                            if future is not None:
                                futures.append(future)
                            current_interval = poll_interval if target is not None \
                                else min(current_interval * 2, maximum_interval)
                            #total_bytes_saved += bytes_saved
                            #if target is None: break
                            #print(len(pool._work_queue.unfinished_tasks))
                            #time.sleep(0)

                    for future in [f for f in futures if f.done()]:
                        futures.remove(future)
                        total_bytes_saved += future.result()
                        logging.debug('CompressionWorker: evaluate joint compression (%d KB)', total_bytes_saved//1000)

                    cls._sleep(instance, current_interval)
                    iteration += 1

        @classmethod
        def _apply_budget(cls, instance, poll_interval=1):
            import vfs.rawcompression

            current_interval, maximum_interval = poll_interval, 32

            while instance.running:
                gop_id, budget_percentage = (instance.database.execute(
                    'SELECT gops.id, (SELECT SUM(size) FROM gops WHERE physical_id = physical_videos.id) / (budget+1.0) FROM gops '
                    'INNER JOIN physical_videos ON gops.physical_id = physical_videos.id '
                    'INNER JOIN logical_videos ON physical_videos.logical_id = logical_videos.id '
                    "WHERE codec = 'rgb' AND "
                    '      joint = 0 AND '
                    '      zstandard IS NULL AND '
                    '      histogram IS NOT NULL AND ' # Metadata creation isn't zstd aware
                    '      (SELECT SUM(size) FROM gops WHERE physical_id = physical_videos.id) > budget '
                    'ORDER BY gops.clock ASC '
                    'LIMIT 1')
                        .fetchone() or [None, None])

                if gop_id is not None and budget_percentage > 0.25:
                    level = max(1, int(round(min(budget_percentage, 1.0) * rawcompression.MAX_LEVEL)))
                    logging.info('StorageBudget: compressing raw GOP %d at level %d', gop_id, level)
                    rawcompression.compress(gop_id, level)
                    current_interval = poll_interval
                else:
                    current_interval = min(current_interval * 2, maximum_interval)

                cls._sleep(instance, current_interval)

        @classmethod
        def _apply_compaction(cls, instance, poll_interval=16):
            current_interval, maximum_interval = poll_interval, 256

            while instance.running:
                left_id, right_id = (instance.database.execute(
                    'SELECT left_physical.id, right_physical.id '
                    'FROM physical_videos left_physical '
                    'INNER JOIN physical_videos right_physical '
                    '  ON left_physical.codec = right_physical.codec AND '
                    '     left_physical.width = right_physical.width AND '
                    '     left_physical.height = right_physical.height AND '
                    '     (SELECT MAX(end_time) FROM gops WHERE physical_id = left_physical.id) = '
                    '     (SELECT MIN(start_time) FROM gops WHERE physical_id = right_physical.id)'
                    'LIMIT 1').fetchone() or [None, None])

                if left_id is not None and right_id is not None:
                    with instance.engine_lock:
                        instance.database.executebatch([
                            "UPDATE gops SET physical_id = %d WHERE physical_id = %d" % (left_id, right_id),
                            "DELETE FROM physical_videos WHERE id = %d" % right_id])
                    logging.info("CompactWorker: Compacted physical video %d into %d" % (right_id, left_id))
                    current_interval = poll_interval
                else:
                    current_interval = min(current_interval * 2, maximum_interval)

                cls._sleep(instance, current_interval)

        @classmethod
        def _apply_compute_error(cls, instance, poll_interval=8):
            from vfs.gop import Gop
            from vfs.videoio import compute_mse

            current_interval, maximum_interval = poll_interval, 256

            while instance.running:
                leaf_gop_id, root_gop_id = instance.database.execute(
                    'WITH RECURSIVE'
                    '  parent(leaf_id, id, parent_id, mse) AS '
                    '    (SELECT id, id, parent_id, mse FROM gops '
                    '       WHERE id = (SELECT id FROM gops WHERE mse IS NULL LIMIT 1)'
                    '    UNION ALL '
                    '    SELECT parent.leaf_id, gops.id, gops.parent_id, gops.mse FROM parent '
                    '    INNER JOIN gops ON parent.parent_id = gops.id)'
                    'SELECT leaf_id, id FROM parent '
                    'WHERE parent_id IS NULL AND mse IS NOT NULL '
                    '').fetchone() or [None, None]

                if leaf_gop_id is not None and root_gop_id is not None:
                    mse = compute_mse(Gop.get(leaf_gop_id), Gop.get(root_gop_id))
                    instance.database.execute('UPDATE gops SET mse = ?, estimated_mse = ? WHERE id = ?', (mse, mse, leaf_gop_id))
                    logging.info("ErrorWorker: Computed precise error for GOP %d", leaf_gop_id)
                    current_interval = poll_interval
                else:
                    current_interval = min(current_interval * 2, maximum_interval)

                cls._sleep(instance, current_interval)

        @classmethod
        def _apply_eviction(cls, instance, poll_interval=1):
            from vfs.gop import Gop
            from vfs.physicalvideo import PhysicalVideo

            current_interval, maximum_interval = poll_interval, 32

            while instance.running:
                logical_id, consumed = (instance.database.execute(
                    'SELECT logical_videos.id, '
                    '       (SELECT SUM(size) FROM gops WHERE physical_id = physical_videos.id) / (budget+1.0) AS consumed '
                    'FROM physical_videos '
                    'INNER JOIN logical_videos ON logical_id = logical_videos.id '
                    #'WHERE consumed > 1 '
                    'ORDER BY consumed DESC '
                    'LIMIT 1').fetchone() or [None, None])

                if logical_id is not None:
                    gop_id, physical_id, cut, _, _, cover_gop_id, gop_count  = (instance.database.execute(
                        'SELECT gop_ranks.id,'
                        '       gop_ranks.physical_id, '
                        '       start_time != min_physical_time AND end_time != max_physical_time AS requires_cut, '
                        '       midpoint_time / physical_duration AS time_percentile, '
                        '       quality_percentile,'
                        '       cover_gop_id, '
                        '       (SELECT COUNT(*) FROM gops gopcount WHERE gopcount.physical_id = gop_ranks.physical_id) AS gop_count '
                        'FROM gop_ranks '
                        'INNER JOIN physical_video_times '
                        '  ON gop_ranks.physical_id = physical_video_times.id '
                        'WHERE physical_video_times.logical_id = ? AND '
                        '      cover_gop_id IS NOT NULL '
                        'ORDER BY clock + ? * (1 - ABS(0.5 - time_percentile)) + ? * (1 - quality_percentile) ASC '
                        'LIMIT 1',
                        (logical_id,
                         instance.temporal_eviction_weight,
                         instance.quality_eviction_weight)).fetchone() or [None, None, None, None, None, None, None])

                    if gop_id is not None:
                        with instance.engine_lock:
                            evict_gop = Gop.get(gop_id)
                            if gop_count == 1:
                                PhysicalVideo.delete(evict_gop.video())
                                logging.info('EvictWorker: Evicting GOP %d (covered by %d; last GOP, also deleting physical video)', gop_id, cover_gop_id)
                            elif cut:
                                new_physical_video = PhysicalVideo.add(
                                    evict_gop.video().logical(),
                                    evict_gop.video().height,
                                    evict_gop.video().width,
                                    evict_gop.video().codec)
                                instance.database.executebatch([
                                    'UPDATE gops SET physical_id = {} WHERE physical_id = {} AND start_time < {}'.format(
                                        new_physical_video.id, physical_id, evict_gop.start_time),
                                    'DELETE FROM gops WHERE id = {}'.format(gop_id)])
                                Gop.delete(evict_gop)
                                logging.info('EvictWorker: Evicting GOP %d (covered by %d; middle GOP, cutting physical video)', gop_id, cover_gop_id)
                            else:
                                logging.info('EvictWorker: Evicting GOP %d (covered by %d; endpoint GOP)', gop_id, cover_gop_id)
                                Gop.delete(evict_gop)

                        current_interval = poll_interval
                else:
                    current_interval = min(current_interval * 2, maximum_interval)

                cls._sleep(instance, current_interval)

    _instance = None
    def __init__(self, *args):
        if VFS._instance is None:
            VFS._instance = VFS.__Singleton(*args)

    def __enter__(self):
        return VFS._instance

    def __exit__(self, *args):
        self.stop()

    @classmethod
    def start(cls, *args):
        if VFS._instance is None:
            VFS._instance = VFS.__Singleton(*args)
            return VFS._instance
        else:
            raise RuntimeError('VFS already started')

    @classmethod
    def stop(cls):
        if not VFS._instance is None:
            VFS._instance.close()
            VFS._instance = None
        else:
            raise RuntimeError('VFS not running')

    @classmethod
    def instance(cls):
        if not cls._instance:
            raise RuntimeError('VFS engine not started')
        else:
            return cls._instance
