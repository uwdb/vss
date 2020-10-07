#ifndef VFS_VIDEOWRITER_H
#define VFS_VIDEOWRITER_H

#include <vector>

namespace vfs {
    class VideoWriter {
    public:
        virtual void write(const std::vector<unsigned char>::iterator&,
                           const std::vector<unsigned char>::iterator&) = 0;
        virtual void flush() = 0;
    };
}

#endif //VFS_VIDEOWRITER_H
