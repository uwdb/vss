#include <fuse.h>
#include "inode.h"
#include "video.h"
#include "virtualvideo.h"

namespace vfs {

VirtualVideo::VirtualVideo(const std::string &name, Video& source, VideoFormat format, size_t height, size_t width,
                           size_t framerate, size_t gop_size, mode_t mode)
        : File(name, source, mode), source_(source), format_(format),
          width_(width), height_(height), framerate_(framerate), gop_size_(gop_size),
          frame_size_(format.frame_size(height, width))
{ }

int VirtualVideo::open(struct fuse_file_info &info) {
    if(info.flags & (O_RDWR | O_APPEND))
        return -EACCES;
    else if(info.flags & O_WRONLY) {
        //std::unique_ptr<Inode> writable = std::make_unique<WritableVirtualVideo>();
        auto &writable = video().substitute(*this, std::make_unique<WritableVirtualVideo>(
                name(), video(), format(), height(), width(), framerate(), gop_size(), mode()));
        return writable.open(info); //video().write(*this, info);
    } else {
        File::open(info);
        video().opened(*this);

        info.fh = (uintptr_t)this;
        return 0;
    }
}


    int read_transcode(const std::filesystem::path &path, char *buffer, size_t size, off_t offset, struct fuse_file_info &info) {
        static std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("../transcode_h264_hevc.sh /tmp/v", "r"), pclose);

        auto read = fread(buffer, 1, size, pipe.get());
        return read;
    }

    int read_raw(const std::filesystem::path &path, char *buffer, size_t size, off_t offset, struct fuse_file_info &info) {
        static std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("../read_raw.sh /tmp/v", "r"), pclose);

        auto read = fread(buffer, 1, size, pipe.get());
        return read;
    }

int read_decode(const std::filesystem::path &path, char *buffer, size_t size, off_t offset, struct fuse_file_info &info) {
    static std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("../decode_h264.sh /tmp/v", "r"), pclose);

    auto read = fread(buffer, 1, size, pipe.get());
    return read;

/*
    static int gop_index = 1;
    static FILE *gop_file = nullptr;
    static GPUContext context{0};
    static VideoLock lock{context};
    static CUVIDFrameQueue queue{lock};
    static DecodeConfiguration configuration{540, 960, 0, 0, {30, 1}, lightdb::Codec::h264()};
    static CudaDecoder decoder{configuration, queue, lock};
    static std::unique_ptr<FileDecodeReader> reader; //{"/home/bhaynes/visualroad/45overlap/panoramic-000-000.mp4"};
    static std::unique_ptr<VideoDecoderSession<DecodeReader::iterator>> session; //{decoder, reader.begin(), reader.end()};
    static char encoded_buffer[1024*1024];
    char gop_filename[255];

    if(reader == nullptr) {
        reader = std::make_unique<FileDecodeReader>("/tmp/v/0.h264");
        session = std::make_unique<VideoDecoderSession<DecodeReader::iterator>>(decoder, reader->begin(), reader->end());
    }

    if(!queue.isEndOfDecode()) {
        auto gpu_frame = session.decode();
        LocalFrame frame(*gpu_frame.cuda());
    }
*/
}

    int VirtualVideo::read(const std::filesystem::path &path, char *buffer, size_t size, off_t offset, struct fuse_file_info &info) {
        static int gop_index = 1;
        static FILE *gop_file = nullptr;
        char gop_filename[255];
        auto source_is_raw = true;
        auto source_transcode = false;

        if(source_is_raw)
            return read_raw(path, buffer, size, offset, info);
            //return read_decode(path, buffer, size, offset, info);
        else if(source_transcode)
            return read_transcode(path, buffer, size, offset, info);

        if(gop_file == nullptr) {
            sprintf(gop_filename, "/tmp/v/%d.h264", gop_index++);
            //std::filesystem::path gop_filename = gop_path / (std::to_string(gop_index++) + ".h264");

            //if(gop_index <= 432)
            if(std::filesystem::exists(gop_filename))
                gop_file = fopen(gop_filename, "rb");
        }

        if(gop_file != nullptr) {
            auto bytes_read = fread(buffer, 1, size, gop_file);
            if(bytes_read == 0) {
                fclose(gop_file);
                gop_file = nullptr;
                return read(path, buffer, size, offset, info);
            } else
                return bytes_read;
        }

        gop_index = 0;
        return 0;
    }

}
