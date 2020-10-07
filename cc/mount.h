#ifndef VFS_MOUNT_H
#define VFS_MOUNT_H

#include <memory>
#include <string>
#include "inode.h"

extern "C" {
    struct fuse_operations;
}

namespace vfs {
    typedef int (*fuse_fill_dir_t) (void*, const char *, const struct stat*, off_t);

    class Mount {
    public:
        explicit Mount(std::string path, std::string mount);
        ~Mount();

        int run();

        Directory &root() { return *root_; }
        const Directory &root() const { return *root_; }
        Inode &find(const std::filesystem::path&);

    private:
        // FUSE callback functions
        int getattr(const std::filesystem::path &path, struct stat *stat);
        int readdir(const std::filesystem::path &path, void*, fuse_fill_dir_t, off_t, struct fuse_file_info&);
        int mkdir(const std::filesystem::path &path, mode_t);
        int mknod(const std::filesystem::path &path, mode_t, dev_t rdev);
        //int mknod(const std::filesystem::path &path, mode_t, struct fuse_file_info&);
        int create(const std::filesystem::path &path, mode_t, struct fuse_file_info&);
        int unlink(const std::filesystem::path &path);
        int open(const std::filesystem::path &path, struct fuse_file_info&);
        int read(const std::filesystem::path &path, char*, size_t, off_t, struct fuse_file_info&);
        int write(const std::filesystem::path &path, const char*, size_t, off_t, struct fuse_file_info&);
        int flush(const std::filesystem::path &path, struct fuse_file_info&);
        int truncate(const std::filesystem::path &path, off_t newsize);
        int readlink(const std::filesystem::path &path, char*, size_t);
        int ioctl(const std::filesystem::path &path, int command, void *argument, struct fuse_file_info&, unsigned int flags, void *data);

        const std::string path_;
        const std::string mount_;
        const std::unique_ptr<struct fuse_operations> operations_;
        std::unique_ptr<Directory> root_;
    };
} // namespace vfs

#endif