#ifndef VFS_VIRTUALVIDEO_H
#define VFS_VIRTUALVIDEO_H

#include "inode.h"
#include "format.h"

namespace vfs {
    class VirtualVideo: public File {
    public:
        VirtualVideo(const std::string &name, Video&, VideoFormat format, size_t height, size_t width,
                     size_t framerate, size_t gop_size, mode_t);

        size_t height() const { return height_; }
        size_t width() const { return width_; }
        const VideoFormat &format() const { return format_; }
        size_t framerate() const { return framerate_; }
        size_t gop_size() const { return gop_size_; }
        Video& video() const { return source_; } //source_.mount().find(source_.path()); }

        int open(struct fuse_file_info&) override;
        int truncate(off_t) override { return EACCES; }
        int read(const std::filesystem::path&, char*, size_t, off_t, struct fuse_file_info&) override;
        int write(const char*, size_t, off_t, struct fuse_file_info&) override { return EACCES; }

    protected:
        const std::optional<size_t> &frame_size() const { return frame_size_; }

    private:
        Video &source_;
        const VideoFormat format_;
        const size_t width_, height_;
        const size_t framerate_, gop_size_;
        const std::optional<size_t> frame_size_;
    };

} // namespace vfs

#endif //VFS_VIRTUALVIDEO_H
