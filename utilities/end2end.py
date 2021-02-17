import concurrent
import os
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor, as_completed

import numpy as np
import cv2

from vfs import engine, api, reconstruction
from vfs.utilities import log_runtime


def load_model():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(540, 540), scale=1 / 255)

    with open("coco.names", "r") as f:
        names = [cname.strip() for cname in f.readlines()]

    return model, names

def inference(model, frame, target_class):
    classes, confidences, boxes = model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
    #hits = []

    for class_id, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        if class_id == target_class:
            x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
            yield confidence, int(y1), int(x1), int(y2), int(x2)
            #hits.append((start_time, confidence, (int(y1), int(x1), int(y2), int(x2))))

    #return hits

def is_dominant_color(frame, target_color, threshold):
    a2D = frame.reshape(-1, frame.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    dominant = np.unravel_index(np.bincount(a1D).argmax(), col_range)

    distance = cv2.norm(dominant - target_color)
    return distance < threshold

def group_by_frames(hits, fps):
    start = None
    hit = None
    for hit in hits:
        if start is None or hit[0] - start > 1/fps:
            if start is not None:
                yield (start, hit[0])
            start = hit[0]

    if start is not None and hit is not None:
        yield (start, hit[0] + 1/fps)

def parse_index_file(index_filename):
    with open(index_filename, "r") as f:
        #for start_time, confidence, box in hits:
        for line in f:
            start_time, confidence, y1, x1, y2, x2 = line.split(',')
            start_time = float(start_time)
            confidence = float(confidence)
            y1 = int(y1)
            x1 = int(x1)
            y2 = int(y2)
            x2 = int(x2)
            yield start_time, confidence, y1, x1, y2, x2

def copy_future_state(source, destination):
    if source.cancelled():
        destination.cancel()
    if not destination.set_running_or_notify_cancel():
        return
    exception = source.exception()
    if exception is not None:
        destination.set_exception(exception)
    else:
        result = source.result()
        destination.set_result(result)

def chain(pool, future, fn, *args):
    result = concurrent.futures.Future()

    def callback(_):
        try:
            temp = pool.submit(fn, future.result(), *args)
            copy = lambda _: copy_future_state(temp, result)
            temp.add_done_callback(copy)
        except:
            result.cancel()
            raise

    future.add_done_callback(callback)
    return result

###########

def ingest_vfs(filename, name=None):
    with engine.VFS(transient=True):
        api.write(name or "v%d" % int(time.time() * 1000), filename)

def index_vfs(name, duration, fps, interval, results_filename):
    model, names = load_model()
    target_class = names.index('car')
    #hits = []
    read_parameters = []
    count = 0
    model_resolution = 540, 540

    with engine.VFS(transient=True), open(results_filename, "w") as f:
        reconstruction.POOL_SIZE = 1

        for index in range(duration * interval):
            start_time = index * (1 / interval)
            end_time = start_time + (1 / fps)
            read_parameters.append((name, None, model_resolution, None, (start_time, end_time), 'rgb', None))

            #if start_time > 30:
            #    break

        #fs = api.readmany2(*zip(*read_parameters), workers=16)
        #afs = as_completed(fs)
        for future in as_completed(api.readmany2i(*zip(*read_parameters), workers=16)):
            with open(future.result(), "rb") as stream:
                frame = np.frombuffer(stream.read(), dtype=np.uint8).reshape(model_resolution[0], model_resolution[1], 3)

            start_time = read_parameters[future.index][4][0]

            for confidence, y1, x1, y2, x2 in inference(model, frame, target_class):
                f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                count += 1

        logging.critical('Index.VFS: %d events indexed', count)
        #return hits

        """
        filenames = api.readmany(*zip(*read_parameters), workers=2)
        #read_parameters = read_parameters[:1]
        #filenames = api.preadmany(read_parameters, workers=4)
        #for filename in filenames:
        for filename in filenames:
            ##with api.read(name, resolution=(540, 704), t=(start_time, end_time), codec='rgb') as filename:
            #filename = api.read(name, resolution=(540,704), t=(start_time, end_time), codec='rgb')
            with open(filename, "rb") as stream:
                #frame = np.fromfile('out.rgb', dtype=np.uint8).reshape((540,704,3))
                frame = np.frombuffer(stream.read(), dtype=np.uint8).reshape(model_resolution[0], model_resolution[1], 3)

            for confidence, y1, x1, y2, x2 in inference(model, frame, target_class):
                f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                count += 1

        logging.critical('Index.VFS: %d events indexed', count)
        #return hits
        """

    """
    model, names = load_model()
    target_class = names.index('car')
    #hits = []
    read_parameters = []

    with engine.VFS(transient=True), open(results_filename, "w") as f:
        for index in range(duration * interval):
            read_parameters.append((name, None, (540, 704), None, (start_time, end_time), 'rgb', None))
            start_time = index * (1 / interval)
            end_time = start_time + (1 / fps)

            #with api.read(name, resolution=(540, 704), t=(start_time, end_time), codec='rgb') as filename:
            filename = api.read(name, resolution=(540,704), t=(start_time, end_time), codec='rgb')
            with open(filename, "rb") as stream:
                #frame = np.fromfile('out.rgb', dtype=np.uint8).reshape((540,704,3))
                frame = np.frombuffer(stream.read(), dtype=np.uint8).reshape(540, 704, 3)

            #hits +=
            for confidence, y1, x1, y2, x2 in inference(model, frame, target_class):
                f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')

            if start_time > 30:
                break
        #return hits
    """

def search_vfs_single(data):
    name, fps, target_color, threshold, start_time, confidence, y1, x1, y2, x2 = data
    end_time = start_time + (1 / fps)
#    prefix = uuid.uuid4().hex

    with engine.VFS(transient=True, child_process=True):
        reconstruction.POOL_SIZE = 1
        #prefix = uuid.uuid4().hex
        #api.read(name, f'out-{prefix}.rgb', t=(start_time, end_time), roi=(y1, x1, y2, x2), codec='rgb')
#        pass

        with api.read(name, t=(start_time, end_time), roi=(y1, x1, y2, x2), codec='rgb') as stream:
        #fn = api.read(name, t=(start_time, end_time), roi=(y1, x1, y2, x2), codec='rgb')
        #with open(fn, 'rb') as stream:
            buffer = np.frombuffer(stream.read(), dtype=np.uint8)
            frame = buffer.reshape(y2 - y1, x2 - x1, 3)
            #except:
            #    pass

        if is_dominant_color(frame, target_color, threshold):
            return start_time, confidence, y1, x1, y2, x2
        else:
            return None #-1, -1, -1, -1, -1, -1
#            f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')

def compute_dominant_color(filename, parameters, target_color, threshold):
    #filename = future.result()
    try:
        with open(filename, "rb") as stream:
            buffer = np.frombuffer(stream.read(), dtype=np.uint8)
            if buffer.size > 0:
                start_time = parameters[4][0]
                y1, x1, y2, x2 = parameters[3]
                confidence = 0
                frame = buffer.reshape(y2 - y1, x2 - x1, 3)

                if is_dominant_color(frame, target_color, threshold):
                    return start_time, confidence, y1, x1, y2, x2
                    #f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                    #count += 1
            return None
    except Exception as e:
        print(filename)
        print(e)


def search_vfs(name, fps, target_color, threshold, index_filename, result_filename):
    count = 0
    read_parameters = []
    futures = []
    for start_time, confidence, y1, x1, y2, x2 in parse_index_file(index_filename):
        read_parameters.append((name, None, None, (y1, x1, y2, x2), (start_time, start_time + (1 / fps)), 'rgb', None))

    #print(len(read_parameters))
    #read_parameters = read_parameters[:16]
    with ProcessPoolExecutor(max_workers=4) as pool, engine.VFS(transient=True, child_process=True):
        for future in api.readmany2i(*zip(*read_parameters), workers=32):
            futures.append(chain(pool, future, compute_dominant_color, read_parameters[future.index], target_color, threshold))
#            futures.append(pool.submit(compute_dominant_color, future, read_parameters[future.index], target_color, threshold))
        #for future in as_completed(api.readmany2i(*zip(*read_parameters), workers=16)):
        #    futures.append(pool.submit(compute_dominant_color, read_parameters[future.index], future.result(), target_color, threshold))

        with open(result_filename, 'w') as f:
            for future in futures: #as_completed(futures):
                if future.result() is not None:
                    start_time, confidence, y1, x1, y2, x2 = future.result()
                    f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                    count += 1

    """
            with open(future.result(), "rb") as stream:
                buffer = np.frombuffer(stream.read(), dtype=np.uint8)

                start_time = read_parameters[future.index][4][0]
                y1, x1, y2, x2 = read_parameters[future.index][3]
                confidence = 0
                frame = buffer.reshape(y2 - y1, x2 - x1, 3)

                if is_dominant_color(frame, target_color, threshold):
                    f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                    count += 1
    """

    logging.critical('Search.VFS: %d events indexed', count)
    return


    """
    read_parameters = []

    for start_time, confidence, y1, x1, y2, x2 in parse_index_file(index_filename):
        read_parameters.append((name, None, None, (y1, x1, y2, x2), (start_time, start_time + (1 / fps)), 'rgb', None))

    with engine.VFS(transient=True), open(result_filename, 'w') as f:
        for i, filename in enumerate(api.readmany(*zip(*read_parameters), workers=16)):
            with open(filename, 'rb') as stream:
                buffer = np.frombuffer(stream.read(), dtype=np.uint8)

                start_time = read_parameters[i][4][0]
                y1, x1, y2, x2 = read_parameters[i][3]
                confidence = 0
                frame = buffer.reshape(y2 - y1, x2 - x1, 3)

                if is_dominant_color(frame, target_color, threshold):
                    f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
    """

    futures = []
    count = 0

    with ProcessPoolExecutor(max_workers=4) as pool, open(result_filename, 'w') as f:
        for data in ((name, fps, target_color, threshold, start_time, confidence, y1, x1, y2, x2) for
         start_time, confidence, y1, x1, y2, x2 in parse_index_file(index_filename)):
            future = pool.submit(search_vfs_single, data)
            futures.append(future)

        #wait(futures)
        for result in (f.result() for f in futures):
            if result is not None:
                start_time, confidence, y1, x1, y2, x2 = result
                f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                count += 1

        logging.critical('Search.VFS: %d events indexed', count)

        #for result in pool.map(search_vfs_single, data):
        #    if result is not None:
        #        start_time, confidence, y1, x1, y2, x2 = result
        #        f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')

    """
    with engine.VFS(transient=True), open(result_filename, 'w') as f:
        for start_time, confidence, y1, x1, y2, x2 in parse_index_file(index_filename):
            end_time = start_time + (1 / fps)
            #y1, x1, y2, x2 = box

            with api.read(name, t=(start_time, end_time), roi=(y1, x1, y2, x2), codec='rgb') as stream:
                frame = np.frombuffer(stream.read(), dtype=np.uint8).reshape(y2 - y1, x2 - x1, 3)

            #a2D = frame.reshape(-1, frame.shape[-1])
            #col_range = (256, 256, 256)  # generically : a2D.max(0)+1
            #a1D = np.ravel_multi_index(a2D.T, col_range)
            #dominant = np.unravel_index(np.bincount(a1D).argmax(), col_range)

            #distance = cv2.norm(dominant - target_color)
            #if distance < threshold:
            if is_dominant_color(frame, target_color, threshold):
                f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                #color_hits.append((start_time, confidence, y1, x1, y2, x2))
            #i += 1
            #if i == 10:
            #    break
        #return color_hits
    """

def stream_vfs(name, resolution, fps, index_filename):
    prefix = uuid.uuid4().hex

    read_parameters = []
    with engine.VFS(transient=True):
        hits = parse_index_file(index_filename)
        for index, (start, end) in enumerate(group_by_frames(hits, fps)):
            read_parameters.append((name, f'vfsout-{prefix}-{index}.mp4', resolution, None, (start, end), 'h264', None))
            #api.read(name, f'out-{prefix}-{index}.mp4', t=(start, end), resolution=resolution, codec='h264')
        wait(list(api.readmany2(*zip(*read_parameters), workers=min(len(read_parameters), 10))))

    #logging.critical('Stream.VFS: %d bytes written', sum(os.path.getsize(p[1]) for p in read_parameters))
###########

def index_fs(filename, duration, fps, interval, results_filename):
    model, names = load_model()
    target_class = names.index('car')
    video = cv2.VideoCapture(filename)
    frame_index = 0
    #hits = []
    count = 0

    with open(results_filename, 'w') as f:
        while frame_index < duration * fps:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video.read()

            #hits += inference(model, frame, target_class, frame_index / fps)
            for confidence, y1, x1, y2, x2 in inference(model, frame, target_class):
                f.write(f'{frame_index / fps},{confidence},{y1},{x1},{y2},{x2}\n')
                count += 1

            frame_index += fps // interval
            #return hits

            #if frame_index / fps > 30:
            #    break

    logging.critical('Index.FS: %d events indexed', count)
        #return hits

def search_fs(filename, fps, target_color, threshold, index_filename, result_filename):
    #color_hits = []
    video = cv2.VideoCapture(filename)
    count = 0

    with open(result_filename, 'w') as f:
        for start_time, confidence, y1, x1, y2, x2 in parse_index_file(index_filename):
            frame_index = start_time * fps
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            #y1, x1, y2, x2 = box

            success, frame = video.read()
            frame = frame[y1:y2, x1:x2, :]

            if is_dominant_color(frame, target_color, threshold):
                f.write(f'{start_time},{confidence},{y1},{x1},{y2},{x2}\n')
                count += 1
                #color_hits.append((start_time, confidence, y1, x1, y2, x2))

        logging.critical('Index.FS: %d events indexed', count)
        #return color_hits

def stream_fs(filename, fps, index_filename):
    prefix = uuid.uuid4().hex
    video = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hits = parse_index_file(index_filename)
    total_size = 0

    for index, (start, end) in enumerate(group_by_frames(hits, fps)):
        out_video = cv2.VideoWriter()
        out_filename = f'out-{prefix}-{index}.mp4'
        out_video.open(out_filename, fourcc, fps, (width, height))

        start_frame_index, end_frame_index = int(start * fps), int(end * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

        for _ in range(start_frame_index, end_frame_index):
            success, frame = video.read()
            if not success:
                assert False
            out_video.write(frame)
        total_size += os.path.getsize(out_filename)

    #logging.critical('Stream.FS: %d bytes written', total_size)

###########

def temp():
    name = 'v'
    threshold = 50
    fps = 30

    with log_runtime('Search.VFS', level=logging.CRITICAL):
        with engine.VFS(transient=True):
            for start_time, confidence, y1, x1, y2, x2 in parse_index_file('index_cars_vfs.csv'):
                end_time = start_time + (1 / fps)
                api.read(name, "foo.mp4", t=(start_time, end_time), roi=(y1, x1, y2, x2), codec='rgb')
                pass
    exit(1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)

    clients = 1
    ingest_filename = "inputs/visualroad-2k-30a.mp4"
    duration = 3600
    fps = 30
    query_iterval = 3
    target_color = np.array([127, 127, 127])
    color_threshold = 50
    resolution = (1080, 1920)

#    temp()

    with engine.VFS(transient=True) as instance:
#        api.vacuum()
        if 'v' not in api.list():
            api.write("v", ingest_filename)

    with ProcessPoolExecutor(max_workers=clients) as pool:
        #with log_runtime('Index.VFS', level=logging.CRITICAL):
        #    index_vfs("v", duration, fps, query_iterval, 'index_cars_vfs.csv')
        #with log_runtime('Search.VFS', level=logging.CRITICAL):
        #    search_vfs("v", fps, target_color, color_threshold, 'index_cars_vfs.csv', 'index_colors_vfs.csv')
        with log_runtime('Stream.VFS', level=logging.CRITICAL):
            stream_vfs('v', resolution, fps, 'index_colors_vfs.csv')

        #with log_runtime('Index.FS', level=logging.CRITICAL):
        #    index_fs(ingest_filename, duration, fps, query_iterval, 'index_cars_fs.csv')
        #with log_runtime('Search.FS', level=logging.CRITICAL):
        #    search_fs(ingest_filename, fps, target_color, color_threshold, 'index_cars_fs.csv', 'index_colors_fs.csv')
        #with log_runtime('Stream.FS', level=logging.CRITICAL):
        #    stream_fs(ingest_filename, fps, 'index_colors_fs.csv')

        #futures = [pool.submit(ingest_vfs, ingest_filename) for _ in range(clients)]
        #futures = [pool.submit(index_vfs, "v1613333241981", 3600, 30, 3) for _ in range(clients)]
        #wait(futures)