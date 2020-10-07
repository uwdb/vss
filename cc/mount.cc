#include <filesystem>
#include <fuse.h>
#include <glog/logging.h>
#include "mount.h"
#include "inode.h"
#include "video.h"
#include "nativevideo.h"

namespace vfs {
    static Mount &instance() {
        return *static_cast<Mount*>(fuse_get_context()->private_data);
    }

    Mount::Mount(std::string path, std::string mount)
        : path_(std::move(path)),
          mount_(std::move(mount)),
          operations_{std::make_unique<fuse_operations>()},
          root_(std::make_unique<RootDirectory>(*this)) {
        root_->emplace_child(std::make_unique<Directory>("dogs", *root_, 00777));
        auto &cats = root_->emplace_child(std::make_unique<Video>("cats", *this, 00777));

        cats.emplace_child(std::make_unique<NativeVideo>("/home/bhaynes/fireworks.hevc", cats, 00777));

        //root_->children().emplace_back(std::make_unique<Directory>("dogs", *root_, 00777));
        //root_->children().emplace_back(std::make_unique<Video>("cats", *this, 00777));

        //auto& cats = dynamic_cast<Video&>(*root_->children().back());
        //cats.children().emplace_back(std::make_unique<NativeVideo>("/home/bhaynes/fireworks.hevc", cats, 00777));
        //, std::vector<std::filesystem::path>{{"/home/bhaynes/foo.mp4"}}));

        operations_->getattr =[](const char *path, struct stat *stat) { return instance().getattr(path, stat); };
        operations_->readdir = [](const char *path, void *buf, fuse_fill_dir_t dir, off_t offset, struct fuse_file_info* fi) { return instance().readdir(path, buf, dir, offset, *fi); };
        operations_->mkdir = [](const char *path, mode_t mode) { return instance().mkdir(path, mode); };
        operations_->mknod = [](const char *path, mode_t mode, dev_t rdev) { return instance().mknod(path, mode, rdev); };
        //operations_->mknod = [](const char *path, mode_t mode, struct fuse_file_info* fi) { return instance().mknod(path, mode, *fi); };
        operations_->create = [](const char *path, mode_t mode, struct fuse_file_info* fi) { return instance().create(path, mode, *fi); };
        operations_->unlink = [](const char *path) { return instance().unlink(path); };
        operations_->open = [](const char *path, struct fuse_file_info* fi) { return instance().open(path, *fi); };
        operations_->read = [](const char *path, char *buf, size_t size, off_t offset, struct fuse_file_info* fi) { return instance().read(path, buf, size, offset, *fi); };
        operations_->write = [](const char *path, const char *buf, size_t size, off_t offset, struct fuse_file_info* fi) { return instance().write(path, buf, size, offset, *fi); };
        operations_->truncate = [](const char *path, off_t newsize) { return instance().truncate(path, newsize); };
        operations_->flush = [](const char *path, struct fuse_file_info* fi) { return instance().flush(path, *fi); };
        operations_->readlink = [](const char *path, char *buf, size_t size) { return instance().readlink(path, buf, size); };
        operations_->ioctl = [](const char *path, int cmd, void *arg, struct fuse_file_info* fi, unsigned int flags, void *data) { return instance().ioctl(path, cmd, arg, *fi, flags, data); };
    }

    Mount::~Mount() = default;

    template<typename T=Inode>
    T &find(std::filesystem::path::iterator &current, const std::filesystem::path::iterator &end, Directory &node) {
        auto child = node.find_child(*current);

        if(current == end && dynamic_cast<T*>(&node) != nullptr)
            return dynamic_cast<T&>(node);
        else if(!child.has_value())
            throw std::runtime_error("ENOENT");
        else if(++current != end)
            return find<T>(current, end, dynamic_cast<Directory&>(child.value().get()));
        else if(dynamic_cast<T*>(&child.value().get()) != nullptr)
            return dynamic_cast<T&>(child.value().get());
        else
            throw std::runtime_error("ENOENT");
    }

    /*
    Inode &find(std::filesystem::path::iterator &current, const std::filesystem::path::iterator &end, Directory &node) {
        auto child = node.find_child(*current);

        if(current == end)
            return node;
        else if(!child.has_value())
            throw std::runtime_error("ENOENT");
        else if(++current != end)
            return find(current, end, dynamic_cast<Directory&>(child.value().get()));
        else
            return child.value();
    }
     */

    template<typename T=Inode>
    int find(const std::filesystem::path &path, Directory &root, T** result) {
        try {
            *result = &find<T>(++path.begin(), path.end(), root);
            return 0;
        } catch(std::runtime_error&) {
            result = nullptr;
            return ENOENT;
        }
    }

    Inode &Mount::find(const std::filesystem::path &path) {
        return vfs::find(++path.begin(), path.end(), root());
    }

    int Mount::getattr(const std::filesystem::path &path, struct stat *stat) {
        LOG(INFO) << "getattr: " << path;
        int result;
        Inode* node;

        if((result = vfs::find(path, root(), &node)) == 0)
            return node->getattr(stat);
        else
            return -result;
    }

    int Mount::readdir(const std::filesystem::path &path, void *buf, vfs::fuse_fill_dir_t filler, off_t offset, struct fuse_file_info& fi) {
        Directory *directory;
        int result;

        LOG(INFO) << "readdir " << path;

        if((result = vfs::find<Directory>(path, root(), &directory)) != 0)
            return -result;
        else
        //else if(node->type() == Inode::directory && (directory = dynamic_cast<Directory*>(node)) != nullptr)
            return directory->readdir(buf, filler, offset, fi);
        //else
          //  return -ENOENT;
    }

    int Mount::mkdir(const std::filesystem::path &path, mode_t mode) {
        int result;
        File* file;

        LOG(INFO) << "create: " << path;

        if((result = vfs::find<File>(path, root(), &file)) == 0)
            return EEXIST;
        else if(result == ENOENT) {
            root_->emplace_child(std::make_unique<Video>(path.filename(), *this, mode));
            return 0;
        } else
            return -result;
    }

    int Mount::mknod(const std::filesystem::path &path, mode_t mode, dev_t rdev) {
        return -EACCES;
    }

    //int Mount::mknod(const std::filesystem::path &path, mode_t mode, struct fuse_file_info& fi) {
    //    return EACCES;
    //}

    int Mount::create(const std::filesystem::path &path, mode_t mode, struct fuse_file_info& info) {
        int result;
        File* file;

        LOG(INFO) << "create: " << path;

        if((result = vfs::find<File>(path, root(), &file)) == 0)
            return EEXIST;
        else if(result == ENOENT) {
            //auto &video =
            //mode |= S_IFDIR;
            //auto &video = root_->emplace_child(std::make_unique<Video>(path.filename(), *this, 00777));
            //video.write(info);
            return -ENOENT;
        } else
            return -result;


        return -EACCES;
    }

    int Mount::unlink(const std::filesystem::path &path) {
        return -1;
    }

    int Mount::open(const std::filesystem::path &path, struct fuse_file_info& info) {
        int result;
        File* file;

        LOG(INFO) << "open: " << path;

        /*if(info.flags & (O_RDWR | O_WRONLY | O_APPEND))
            return EACCES;
        else */if((result = vfs::find<File>(path, root(), &file)) == 0)
            return file->open(info);
        else
            return -result;
    }

    int Mount::read(const std::filesystem::path &path, char *buffer, size_t size, off_t offset, struct fuse_file_info& info) {
        int result;
        File* node;

        LOG(INFO) << "read: " << path;

        File *file = reinterpret_cast<File*>(info.fh);
        return file->read(path, buffer, size, offset, info);

        /*if((result = vfs::find(path, root(), &node)) == 0)
            return dynamic_cast<File*>(node)->open(info);
        else
            return -result;


        if(offset < 5) {
            memcpy(buf, "A", 2);
            return 1;
        }
        return 0;*/
    }

    int Mount::write(const std::filesystem::path &path, const char *buffer, size_t size, off_t offset, struct fuse_file_info& info) {
        int result;
        File* file;

        LOG(INFO) << "truncate: " << path;

        /*if(newsize != 0)
            return -EACCES;
        else*/ if((result = vfs::find<File>(path, root(), &file)) == 0)
            return file->write(buffer, size, offset, info);
        else
            return -result;
    }

    int Mount::flush(const std::filesystem::path &path, struct fuse_file_info &info) {
        int result;
        File* file;

        LOG(INFO) << "flush: " << path;

        if((result = vfs::find<File>(path, root(), &file)) == 0)
            return file->flush(info);
        else
            return -result;
    }

    int Mount::truncate(const std::filesystem::path &path, off_t newsize) {
        int result;
        File* file;

        LOG(INFO) << "truncate: " << path;

        /*if(newsize != 0)
            return -EACCES;
        else*/ if((result = vfs::find<File>(path, root(), &file)) == 0)
            return file->truncate(newsize);
        else
            return -result;
    }

    int Mount::readlink(const std::filesystem::path &path, char *buffer, size_t size) {
        int result;
        Inode* node;

        LOG(INFO) << "readlink: " << path;

        if((result = vfs::find(path, root(), &node)) == 0)
            return dynamic_cast<Link*>(node)->readlink(buffer, size);
        else
            return -result;
    }

    int Mount::ioctl(const std::filesystem::path &path, int cmd, void *arg, struct fuse_file_info& fi, unsigned int flags, void *data) {
        return -ENOTTY;
    }


    int Mount::run() {
        char *argv[] {const_cast<char*>(path_.c_str()),
                      const_cast<char*>(mount_.c_str()),
                      "-obig_writes",
                      "-olarge_read",
                      "-omax_read=1048576",
                      "-omax_readahead=1048576",
                      "-f"};
        return fuse_main(std::size(argv), argv, &*operations_, this);
    }
}
