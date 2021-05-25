import time
import os
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import logging

from vfs import rawcompression, eviction
from vfs.db import Database

DEFAULT_PERFECT_QUALITY_THRESHOLD = 100
DEFAULT_TEMPORAL_EVICTION_WEIGHT = 4
DEFAULT_QUALITY_EVICTION_WEIGHT = 2
DEFAULT_EVICTION_POLICY = eviction.EVICTION_POLICY_VSS

def _create_metadata_process(gop_id, database_filename, gop_filename, height, width, codec):
    from vfs.physicalvideo import PhysicalVideo
    from vfs.histogram import Histogram
    from vfs.descriptor import Descriptor

    database = Database(None, database_filename, child_process=True)
    histogram, frame = Histogram.create(gop_filename, (height, width, PhysicalVideo.CHANNELS), codec)
    keypoints, descriptors = Descriptor.create(frame)
    database.execute(
        'UPDATE gops SET histogram = ?, keypoints = ?, descriptors = ? WHERE id = ?',
        (histogram, keypoints, descriptors, gop_id))

def _evict_process(database_filename, eviction_policy, temporal_eviction_weight, quality_eviction_weight):
    db = Database(None, database_filename, child_process=True)
    return eviction.get_gop_eviction_candidate(db, eviction_policy, temporal_eviction_weight, quality_eviction_weight)

class VFS(object):
    class __Singleton(object):
        DEFAULT_PATH = 'data'
        DATABASE_NAME = 'vfs.db'

        def __init__(self,
                     transient=False,
                     child_process=False,
                     create_metadata=True,
                     apply_joint_compression=True,
                     apply_deferred_compression=True,
                     apply_compaction=True,
                     verify_quality=True,
                     apply_eviction=True,
                     path=DEFAULT_PATH,
                     database_filename=os.path.join(DEFAULT_PATH, DATABASE_NAME),
                     temporal_eviction_weight=DEFAULT_TEMPORAL_EVICTION_WEIGHT,
                     quality_eviction_weight=DEFAULT_QUALITY_EVICTION_WEIGHT,
                     perfect_quality_threshold=DEFAULT_PERFECT_QUALITY_THRESHOLD,
                     eviction_policy=DEFAULT_EVICTION_POLICY):
            from vfs.jointcompression import JointCompression

            logging.debug('VFS: data path "%s"', path)
            logging.debug('VFS: database "%s"', database_filename)

            self.path = path
            self.perfect_quality_threshold = perfect_quality_threshold
            self.temporal_eviction_weight = temporal_eviction_weight
            self.quality_eviction_weight = quality_eviction_weight
            self.clock = 0
            self.engine_lock = threading.Lock()
            self.database = Database(self, database_filename, child_process=child_process)
            self.compression = JointCompression()
            self.eviction_policy = eviction_policy

            if not transient and not os.path.exists(self.path):
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
                    if create_metadata: self._metadata_worker.start()
                    if apply_joint_compression: self._compression_worker.start()
                    if apply_deferred_compression: self._budget_worker.start()
                    if apply_compaction: self._compaction_worker.start()
                    if verify_quality: self._error_worker.start()
                    if apply_eviction: self._eviction_worker.start()
            else:
                logging.debug('VFS: engine already running')

        def close(self):
            self.running = False
            if not self.transient:
                logging.info('VFS: stopping engine')
                if self._metadata_worker.is_alive(): self._metadata_worker.join()
                if self._compression_worker.is_alive(): self._compression_worker.join()
                if self._budget_worker.is_alive(): self._budget_worker.join()
                if self._compaction_worker.is_alive(): self._compaction_worker.join()
                if self._error_worker.is_alive(): self._error_worker.join()
                if self._eviction_worker.is_alive(): self._eviction_worker.join()
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
            last_physical_id, last_gop_id = -1, -1
            current_interval = poll_interval
            futures = {}

            with ProcessPoolExecutor(max_workers=pool_size) as pool:
                while instance.running:
                    if len(futures) < pool_size:
                        physical_id, gop_id, filename, height, width, codec = (instance.database.execute(
                            'SELECT physical_id, gops.id, physical_videos.filename, height, width, codec FROM gops '
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

                            futures[gop_id] = pool.submit(_create_metadata_process, instance.database.filename, filename, gop_id, height, width, codec)
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

            return
            pool_size = 1 # TODO TODO

            #cluster_period = 1
            futures = []

            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                while instance.running:
                    if iteration % cluster_period == 0 and PhysicalVideo.count() > 0: # and instance.database.execute(
                            #'SELECT 1 FROM gops WHERE histogram IS NULL LIMIT 1').fetchone() is None:s                      logging.info("CompressionWorker: reclustering")
                        clusters = Histogram.cluster_all()
                        if instance.database.execute('SELECT 1 FROM gops '
                                                     'WHERE examined <= ? AND '
                                                     '      physical_id < (SELECT MAX(id) FROM physical_videos) LIMIT 1', epoch)\
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
            current_interval, maximum_interval = poll_interval, 32

            with ProcessPoolExecutor(max_workers=1) as pool:
                while instance.running:
                    future = pool.submit(_evict_process, instance.database.filename, instance.eviction_policy, instance.temporal_eviction_weight, instance.quality_eviction_weight)
                    gop_id, cut = future.result()
                    if gop_id is not None:
                        with VFS._instance.engine_lock:
                            eviction.evict_gop(instance.database, gop_id, cut)
                        current_interval = poll_interval
                    else:
                        current_interval = min(current_interval * 2, maximum_interval)

                    cls._sleep(instance, current_interval)

    _instance = None
    def __init__(self, *args, **kwargs):
        if VFS._instance is None:
            VFS._instance = VFS.__Singleton(*args, **kwargs)

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
        if VFS._instance is None:
            raise RuntimeError('VFS not running')
        elif VFS._instance.transient:
            pass
        else:
            with VFS._instance.engine_lock:
                VFS._instance.close()
                VFS._instance = None

    @classmethod
    def instance(cls):
        if not cls._instance:
            raise RuntimeError('VFS engine not started')
        else:
            return cls._instance
