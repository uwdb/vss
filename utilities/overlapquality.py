import os
import numpy as np
import cv2
from skimage.measure import compare_ssim
from vfs import api
from vfs import utilities
from vfs import logicalvideo
from vfs import engine
from vfs import videoio
from vfs import jointcompression

def evaluate_quality_overlap_right(gop, reference_directory):
    source_filename = 'out_decompress.mp4'
    jointcompression.JointCompression.co_decompress(gop, source_filename)
    original_filename = engine.VFS.instance().database.execute("SELECT original_filename from gops WHERE id = ?", gop.id).fetchone()[0]
    right_filename = gop.filename.format('right')

    separate_shape = videoio.get_shape(right_filename)

    source_video = cv2.VideoCapture(source_filename)
    reference_video = cv2.VideoCapture(os.path.join(reference_directory, os.path.basename(original_filename)))

    separate_psnr, overlap_psnr = 0, 0
    count = 0

    while reference_video.isOpened():
        sret, source_frame = source_video.read()
        rret, reference_frame = reference_video.read()

        if not sret or not rret:
            break

        count += 1

        source_separate = source_frame[:, -separate_shape[1]:]
        ref_separate = reference_frame[:, -separate_shape[1]:]
        source_overlap = source_frame[:, separate_shape[1]:]
        ref_overlap = reference_frame[:, separate_shape[1]:]

        separate_psnr += utilities.psnr(ref_separate, source_separate)
        overlap_psnr += utilities.psnr(ref_overlap, source_overlap)

        #cv2.imwrite('source.png', source_frame)
        #cv2.imwrite('separate.png', source_separate)
        #cv2.imwrite('refsep.png', ref_separate)
        #cv2.imwrite('reference.png', reference_frame)

    print(f'GOP {gop.id} separate: {separate_psnr//count}, overlap: {overlap_psnr//count}')
    return separate_psnr, overlap_psnr, count

def evaluate_quality_overlap_left(gop, reference_directory):
    source_filename = 'out_decompress.mp4'
    overlap_filename_template = gop.filename
    left_filename = overlap_filename_template.format('left')

    jointcompression.JointCompression.co_decompress(gop, source_filename)

    separate_shape = videoio.get_shape(left_filename)

    source_video = cv2.VideoCapture(source_filename)
    reference_video = cv2.VideoCapture(os.path.join(reference_directory, os.path.basename(overlap_filename_template.replace('-{}', ''))))

    separate_psnr, overlap_psnr = 0, 0
    count = 0

    while reference_video.isOpened():
        sret, source_frame = source_video.read()
        rret, reference_frame = reference_video.read()

        if not sret or not rret:
            break

        count += 1

        source_separate = source_frame[:, :separate_shape[1]]
        ref_separate = reference_frame[:, :separate_shape[1]]
        source_overlap = source_frame[:, separate_shape[1]:]
                         #overlap_shape[0] // 2 - separate_shape[0]//2:overlap_shape[0] // 2 + separate_shape[0]//2, overlap_shape[1]:]
        ref_overlap = reference_frame[:, separate_shape[1]:]
                      #overlap_shape[0] // 2 - separate_shape[0]//2:overlap_shape[0] // 2 + separate_shape[0]//2, overlap_shape[1]:]

        separate_psnr += utilities.psnr(ref_separate, source_separate)
        overlap_psnr += utilities.psnr(ref_overlap, source_overlap)

    print(f'GOP {gop.id} separate: {separate_psnr//count}, overlap: {overlap_psnr//count}')
    return separate_psnr, overlap_psnr, count


def evaluate_quality_separate(gop, source_filename, reference_directory):
    source_video = cv2.VideoCapture(source_filename)
    reference_video = cv2.VideoCapture(os.path.join(reference_directory, os.path.basename(source_filename)))

    psnr, count = 0, 0

    while source_video.isOpened():
        ret, frame = source_video.read()
        rret, reference_frame = reference_video.read()

        if not ret:
            break

        count += 1
        current_psnr = utilities.psnr(frame, reference_frame)
        psnr += current_psnr
        print(f'GOP {gop.id} frame {count} separate: {current_psnr}')
        #if current_psnr > 300:
        #    cv2.imwrite('good.png', frame)
        #    cv2.imwrite('good_ref.png', reference_frame)


    print(f'GOP {gop.id} separate: {psnr//count}')
    return psnr, count


def evaluate_ssim(name, reference_filename, source_filename=None):
    if source_filename is None:
        source_filename = 'out_filename.mp4'
        api.read(name, source_filename)
        #api.read(name, out_filename)
        #api.read(right_name, right_filename)

    source_video = cv2.VideoCapture(source_filename)
    reference_video = cv2.VideoCapture(reference_filename)
    ssim = []
    count = 0

    while reference_video.isOpened():
        sret, source = source_video.read()
        rret, reference = reference_video.read()

        if not rret:
            break

        count += 1
        ssim.append(compare_ssim(source, reference, multichannel=True))
        print(f'SSIM frame {len(ssim)} mean {sum(ssim)/len(ssim)}')
        #cv2.imwrite(f'ssim_left_{count}.png', source)
        #cv2.imwrite(f'ssim_right_{count}.png', reference)

    print('SSIM %d' % (sum(ssim) / len(ssim)))
    return sum(ssim) / len(ssim)

def evaluate_quality(left_name, right_name, reference_directory, reference_left_filename, reference_right_filename):
    with engine.VFS():
        logical_left = logicalvideo.LogicalVideo.get_by_name(left_name)
        physical_left = list(logical_left.videos())[0]
        gops_left = physical_left.gops()

        separate_psnr, overlap_psnr = 0, 0
        separate_count, overlap_count = 0, 0

        for gop in gops_left:
            if '{}' not in gop.filename:
                left_separate_psnr, left_count = evaluate_quality_separate(gop, gop.filename, reference_directory)
                separate_psnr += left_separate_psnr
                separate_count += left_count
                pass
            else:
                left_separate_psnr, left_overlap_psnr, left_count = evaluate_quality_overlap_left(gop, reference_directory)
                separate_psnr += left_separate_psnr
                overlap_psnr += left_overlap_psnr
                overlap_count += left_count
                separate_count += left_count
                pass

        logical_right = logicalvideo.LogicalVideo.get_by_name(right_name)
        physical_right = list(logical_right.videos())[0]
        gops_right = physical_right.gops()

        for gop in gops_right:
            if '{}' not in gop.filename:
                right_separate_psnr, right_count = evaluate_quality_separate(gop, gop.filename, reference_directory)
                separate_psnr += right_separate_psnr
                separate_count += right_count
                pass
            else:
                right_separate_psnr, right_overlap_psnr, right_count = evaluate_quality_overlap_right(gop, reference_directory)
                separate_psnr += right_separate_psnr
                overlap_psnr += right_overlap_psnr
                overlap_count += right_count
                separate_count += right_count

        print(f'Overall separate: {separate_psnr // separate_count}, overlap: {overlap_psnr // overlap_count}')

        #ssim_left = evaluate_ssim(left_name, reference_left_filename, source_filename='out_left.mp4')
        ssim_right = evaluate_ssim(right_name, reference_right_filename, source_filename='out_right.mp4')


        return separate_psnr / separate_count, overlap_psnr / overlap_count, ssim_left, ssim_right


def evaluate_qualityXXX(overlap_tilename_template, left_reference_filename, right_reference_filename):
    left_filename = overlap_tilename_template.format('left')
    overlap_filename = overlap_tilename_template.format('overlap')
    right_filename = overlap_tilename_template.format('right')

    left = cv2.VideoCapture(left_filename)
    right = cv2.VideoCapture(right_filename)
    overlap = cv2.VideoCapture(overlap_filename)
    reference_left = cv2.VideoCapture(left_reference_filename)
    reference_right = cv2.VideoCapture(right_reference_filename)

    left_separate_psnr, right_separate_psnr, left_overlap_psnr = 0, 0, 0
    count = 0

    while reference_left.isOpened():
        lret, left_frame = left.read()
        rret, right_frame = right.read()
        oret, overlap_frame = overlap.read()
        rlret, reference_left_frame = reference_left.read()
        rrret, reference_right_frame = reference_right.read()

        if not rlret:
            break

        count += 1
        ref_left = reference_left_frame[:, :left_frame.shape[1]]
        ref_right = reference_right_frame[:, -right_frame.shape[1]:]
        #print(reference_right_frame.shape)
        #print(right_frame.shape)

        overlap_recovered_left = overlap_frame[overlap_frame.shape[0] // 2 - reference_left_frame.shape[0] // 2:overlap_frame.shape[0] // 2 + reference_left_frame.shape[0] // 2,
                                               :reference_left_frame.shape[1] - left_frame.shape[1]]
        ref_overlap_left = reference_left_frame[:, -overlap_recovered_left.shape[1]:]
        recovered_full_left = np.hstack([left_frame, overlap_recovered_left])
        #print(recovered_left.shape, reference_left_frame.shape)
        #print(reference_left_frame.shape)
        #print(left_frame.shape)
        #print(overlap_frame[overlap_frame.shape[0] // 2 - reference_left_frame.shape[0] // 2:overlap_frame.shape[0] // 2 + reference_left_frame.shape[0] // 2, :reference_left_frame.shape[0] - left_frame.shape[0]].shape)
        #print(recovered_full_left.shape)

        #print(overlap_recovered_left.shape, ref_overlap_left.shape)
        #print(utilities.psnr(ref_overlap_left, overlap_recovered_left))
        #cv2.imwrite('rol.png', ref_overlap_left)
        #cv2.imwrite('orl.png', overlap_recovered_left)

        left_separate_psnr += utilities.psnr(left_frame, ref_left)
        right_separate_psnr += utilities.psnr(right_frame, ref_right)
        left_overlap_psnr += utilities.psnr(ref_overlap_left, overlap_recovered_left)
#        print(ref_left.shape == left_frame.shape, ref_right.shape == right_frame.shape)
#        print(ref_right.shape, right_frame.shape)
        #print(utilities.psnr(right_frame, ref_right))
        if utilities.psnr(right_frame, ref_right) < 31:
            #print(ref_right.shape)
            cv2.imwrite('refright.png', ref_right)
            cv2.imwrite('right.png', right_frame)

    return left_separate_psnr / count, right_separate_psnr / count, left_overlap_psnr / count

if __name__ == '__main__':
    print(evaluate_quality('l', 'r', 'databak', '/home/bhaynes/cidr/figure4/panos/p0.mp4', '/home/bhaynes/cidr/figure4/panos/p30.mp4'))
    #print(evaluate_quality('data/l-1-0002-{}.h264', 'databak/l-1-0002.h264', 'databak/r-2-0002.h264'))