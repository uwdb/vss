#include <sys/stat.h>
#include <fuse.h>
#include <glog/logging.h>
#include "mount.h"
#include "inode.h"
#include "video.h"

namespace vfs {

Inode::Inode(std::string name, const Mount &mount, const Directory &parent, mode_t mode)
    : name_(std::move(name)), path_(parent.path() / name), mount_(mount), parent_(parent), mode_(mode)
{ }

Inode::Inode(const std::string &name, const Directory &parent, mode_t mode)
    : Inode(name, parent.mount(), parent, mode)
{ }

//Error::Error(const Mount& mount, int error)
//        : error_(error), Inode({}, mount, mount.root(), Inode::error, 0)
//{ }


int Directory::getattr(struct stat *stat) {
    stat->st_mode = S_IFDIR | mode_;
    stat->st_nlink = 2 + directories_;
    return 0;
}

int Directory::readdir(void *buffer, vfs::fuse_fill_dir_t filler, off_t offset, struct fuse_file_info &info) {
    info.keep_cache = 1;
    filler(buffer, ".", nullptr, 0);
    filler(buffer, "..", nullptr, 0);
    std::for_each(children().begin(), children().end(),[&filler, &buffer](auto &child) {
        filler(buffer, child->name().c_str(), nullptr, 0); });
    //for (auto it = children_.begin(); it != children_.end(); ++it)
    //    filler(buffer, it->name().c_str(), nullptr, 0);
    return 0;
}

//    Video::Video(const std::string &name, const Mount &mount, mode_t mode) //, std::vector<std::filesystem::path> sources)
  //          : Directory(name, mount.root(), mode)  //, sources_(std::move(sources))
    //{ }

  //  VirtualVideo::VirtualVideo(const std::string &name, const Video &source, const size_t width, const size_t height, const mode_t mode)
//            : File(name, source, mode), source_(source), width_(width), height_(height)
    //{ }




    /*
    int Video::getattr(struct stat *stat) {
        stat->st_mode = S_IFDIR | mode_;
        stat->st_nlink = 2 + directories_;
        return 0;
    }

    int Video::readdir(void *buffer, vfs::fuse_fill_dir_t filler, off_t offset, struct fuse_file_info* fi) {
        filler(buffer, ".", nullptr, 0);
        filler(buffer, "..", nullptr, 0);
        //filler(buffer, "native", nullptr, 0);
        std::for_each(children().begin(), children().end(), [&filler, &buffer](auto &child) {
            filler(buffer, child->name().c_str(), nullptr, 0); });
        //for (auto it = children_.begin(); it != children_.end(); ++it)
        //    filler(buffer, it->name().c_str(), nullptr, 0);
        return 0;
    }
*/

    int File::getattr(struct stat *stat) {
        stat->st_mode = S_IFREG | 0444;
        stat->st_nlink = 1;
        stat->st_size = 1; //size();
        return 0;
    }

    int File::open(struct fuse_file_info &info) {
        //printf("open flags: %d\n", info.flags);
        info.fh = (uintptr_t)this;
        info.nonseekable = 1;
        info.direct_io = 1;
        return 0;
    }

    int Link::getattr(struct stat *stat) {
        stat->st_mode = S_IFLNK | 0777;
        stat->st_nlink = 1;
        stat->st_size = path_.string().size();
        return 0;
    }

    int Link::readlink(char *buf, size_t size) {
        strncpy(buf, path_.c_str(), size);
        return 0;
    }
} // namespace vfs
