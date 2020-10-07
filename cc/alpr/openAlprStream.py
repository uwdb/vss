import sys
from alprstream import AlprStream
from openalpr import Alpr
import cv2
from multiprocessing.pool import ThreadPool


def feed_alpr_stream(vid_path, alpr_stream, queue_size, width, height):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    num_read_frames = 0
    while ret:
        while alpr_stream.get_queue_size() == queue_size:
            pass

        alpr_stream.push_frame(frame, 3, width, height)
        ret, frame = cap.read()
        num_read_frames += 1
    print("Number of frames fed to alpr_stream: %d" % num_read_frames)


if __name__ == '__main__':
    alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data", "SEpKS0xNTk3i6eP66uD12Ov61fzt/ubm/vbm/KWgoaWhoKiipayvqpeTm5qWnJ+WAMA8vVYIl6+7NZfurh9QM9uFszy3FXCs/jyHeXmSeY5Ycrv5wctus78Rhc0jMCepWAundOt6bPrdUbawPiB1TkI+WqZtcgufOfj4GD4+Uk/YcS4KXqtaKy8+rqYuo0Yi")
    if not alpr.is_loaded():
        print("Error loading Alpr")
        sys.exit(1)

    queue_size = 10
    alpr_stream = AlprStream(frame_queue_size=queue_size, use_motion_detection=True)
    if not alpr_stream.is_loaded():
        print("Error loading AlprStream")

    vid_path = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    output_file = sys.argv[4]

    pool = ThreadPool(2)
    apply_result = pool.apply_async(feed_alpr_stream, [vid_path, alpr_stream, queue_size, width, height])

    with open(output_file, 'w') as output:
        while alpr_stream.get_queue_size() > 0 or not apply_result.ready():
            single_frame = alpr_stream.process_frame(alpr)
            active_groups = len(alpr_stream.peek_active_groups())
            # print('Active groups: {:<3} \tQueue size: {}'.format(active_groups, alpr_stream.get_queue_size()))
            groups = alpr_stream.pop_completed_groups()
            for group in groups:
                print('=' * 40)
                print('Group from frames {}-{}'.format(group['frame_start'], group['frame_end']))
                print('Plate: {} ({:.2f}%)'.format(group['best_plate']['plate'], group['best_plate']['confidence']))
                print('=' * 40)

                start_frame = group['frame_start']
                end_frame = group['frame_end']
                for frame in range(start_frame, end_frame+1):
                    output.write(str(frame) + "\n")

    pool.close()
    pool.join()

    alpr.unload()
