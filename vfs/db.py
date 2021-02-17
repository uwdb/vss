import logging
import sqlite3
import numpy as np
import io
import os

def _adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def _convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

class QueryResult(object):
    def __init__(self, resultset):
        self.resultset = resultset

    def close(self):
        self.resultset.close()
        self.resultset.connection.close()

    def fetchone(self):
        data = self.resultset.fetchone()
        self.close()
        return data

    def fetchall(self):
        data = self.resultset.fetchall()
        self.close()
        return data

    def fetchmany(self, n):
        data = self.resultset.fetchmany(n)
        self.close()
        return data

    @property
    def lastrowid(self):
        rowid = self.resultset.lastrowid
        self.close()
        return rowid


class Database(object):
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("NUMPY", _convert_array)

    @classmethod
    def clear(cls):
        if os.path.exists('data/vfs.db'):
            os.remove('data/vfs.db')

    def __init__(self, engine, filename, child_process=False):
        #TODO
        self.filename = filename

        connection = self.get_connection()
        cursor = connection.cursor()

        # Skip schema creation in a child process
        if child_process:
            return

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logical_videos (
                id integer PRIMARY KEY AUTOINCREMENT,
                name varchar(50) UNIQUE NOT NULL,
                budget float NOT NULL DEFAULT 0
                );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS physical_videos (
                id integer PRIMARY KEY AUTOINCREMENT,
                logical_id integer NOT NULL REFERENCES logical_videos(id),
                width int NOT NULL,
                height int NOT NULL,
                codec varchar(4) NOT NULL
                );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gops (
                id integer PRIMARY KEY AUTOINCREMENT,
                physical_id integer NOT NULL REFERENCES physical_videos(id),
                filename varchar(50) NOT NULL,
                start_time float NOT NULL,
                cluster_id int DEFAULT 0 NOT NULL,
                joint bool NOT NULL DEFAULT FALSE,
                examined int NOT NULL DEFAULT 0,
                histogram NUMPY,
                descriptors NUMPY,
                keypoints NUMPY,
                homography NUMPY,
                shapes NUMPY,
                is_left bool NOT NULL DEFAULT False,
                size integer NOT NULL DEFAULT 0,
                zstandard integer,
                clock integer DEFAULT 0,
                end_time float NOT NULL DEFAULT 0,
                fps integer NOT NULL,
                mse integer,
                estimated_mse integer,
                parent_id integer REFERENCES gops(id),
                original_filename varchar(50),
                original_size integer NOT NULL
                );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gop_joint_aborted (
                gop1 integer NOT NULL REFERENCES gops(id),
                gop2 integer NOT NULL REFERENCES gops(id)
                );
        ''')

        #cursor.execute('ALTER TABLE gops ADD COLUMN original_size integer DEFAULT 0')

#                   COALESCE((SELECT SUM(end_time - start_time)
#                              FROM gops previous
#                              WHERE previous.physical_id = gops.physical_id AND
#                                    previous.start_time < gops.start_time), 0) AS absolute_start_time,
        cursor.execute('DROP VIEW IF EXISTS gop_ranks') # Reset perfect quality threshold
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS gop_ranks
            AS
                SELECT
                    physical_videos.logical_id,
                    gops.physical_id,
                    gops.id,
                    gops.clock,
                    gops.start_time,
                    gops.start_time != min_physical_time AND gops.end_time != max_physical_time AS requires_cut, 
                    gops.end_time - gops.start_time AS gop_duration,
                    COUNT(lower_quality.id) AS lower_quality_count
            FROM gops
            INNER JOIN physical_video_times physical_videos 
                ON gops.physical_id = physical_videos.id
            LEFT JOIN gops lower_quality
                ON lower_quality.physical_id = gops.physical_id AND 
                   lower_quality.estimated_mse < gops.estimated_mse
            WHERE EXISTS(
                   SELECT id FROM gops cover 
                   WHERE cover.physical_id IN (SELECT id FROM physical_videos cover_physical 
                                               WHERE logical_id = physical_videos.logical_id) AND 
                         cover.start_time <= gops.start_time AND 
                         cover.end_time >= gops.end_time AND 
                         cover.id != gops.id AND
                         cover.estimated_mse <= max(gops.estimated_mse, {}))
            GROUP BY physical_videos.logical_id, gops.physical_id, gops.id, gops.clock, gops.start_time, gop_duration, requires_cut
            LIMIT 2500
        '''.format(engine.perfect_quality_threshold))

        cursor.execute('DROP VIEW IF EXISTS old_gop_ranks') # Reset perfect quality threshold
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS old_gop_ranks
            AS
                SELECT
                    *,
                    (end_time - start_time) / 2 AS midpoint_time,
                    CAST((SELECT COUNT(*)
                     FROM gops lower_quality 
                     WHERE lower_quality.physical_id = gops.physical_id AND 
                           lower_quality.estimated_mse < gops.estimated_mse) AS float)
                     / (SELECT COUNT(*) FROM gops all_gops WHERE all_gops.physical_id = gops.physical_id) AS quality_percentile,
                    (SELECT cover.id FROM gops cover 
                           INNER JOIN physical_videos physical_cover ON physical_cover.id = cover.physical_id
                           WHERE physical_videos.logical_id = physical_cover.logical_id AND 
                                 cover.start_time <= gops.start_time AND 
                                 cover.end_time >= gops.end_time AND 
                                 cover.id != gops.id AND
                                 cover.estimated_mse <= max(gops.estimated_mse, {})) AS cover_gop_id
            FROM gops
            INNER JOIN physical_videos ON gops.physical_id = physical_videos.id
        '''.format(engine.perfect_quality_threshold))

        cursor.execute('''
            CREATE VIEW IF NOT EXISTS physical_video_times
            AS
                SELECT
                    *,
                    (SELECT MIN(start_time) FROM gops WHERE gops.physical_id = physical_videos.id) AS min_physical_time,
                    (SELECT MAX(end_time) FROM gops WHERE gops.physical_id = physical_videos.id) AS max_physical_time,
                    (SELECT MAX(end_time) FROM gops WHERE gops.physical_id = physical_videos.id) -
                      (SELECT MIN(start_time) FROM gops WHERE gops.physical_id = physical_videos.id) AS physical_duration
            FROM physical_videos
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_time
            ON gops(start_time)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_examined
            ON gops(examined)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_cluster
            ON gops(cluster_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_mismatches
            ON gop_joint_aborted(gop1)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_estimated_mse
            ON gops(estimated_mse)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_start_time
            ON gops(start_time)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_end_time
            ON gops(end_time)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS gop_physical_id
            ON gops(physical_id)
        ''')

        connection.commit()
        cursor.close()
        connection.close()

    def get_connection(self):
        return sqlite3.connect(self.filename, timeout=120, detect_types=sqlite3.PARSE_DECLTYPES)

    def execute(self, sql, *args):
        #print(sql)
        if args is not None and len(args) == 1 and type(args[0]) is not tuple:
            args = (args,)
        connection = self.get_connection()
        cursor = connection.cursor()
        #logging.debug(sql)
        result = QueryResult(cursor.execute(sql, *args))
        connection.commit()
        return result

    def executebatch(self, sql_batch, *args):
        if args is not None and len(args) == 1 and type(args[0]) is not tuple:
            args = (args,)

        connection = self.get_connection()
        cursor = connection.cursor()
        for sql in sql_batch:
            #logging.debug(sql)
            cursor.execute(sql, *args)
        cursor.close()
        connection.commit()
        connection.close()
