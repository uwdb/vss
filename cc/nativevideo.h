#ifndef VFS_NATIVEVIDEO_H
#define VFS_NATIVEVIDEO_H

#include "inode.h"

namespace vfs {

    class NativeVideo : public Link {
    public:
        NativeVideo(const std::filesystem::path &path, const Directory &directory, const mode_t mode)
                : Link("1920x1080.hevc", path, directory, mode)
        { }
    };

    /*class WritableVideo : public NativeVideo {
    public:
        WritableVideo(const std::filesystem::path &path, const Directory &directory, const mode_t mode)
                : NativeVideo(CreateEmptyVideo(path.extension()), directory, mode)
        { }

    private:
        std::string CreateEmptyVideo(const std::string &extension) {
            return "writable.hevc";
        }
    };*/

} // namespace vfs

#endif //VFS_NATIVEVIDEO_H
