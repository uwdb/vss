#ifndef VFS_INODE_H
#define VFS_INODE_H

#include <sys/stat.h>
#include <algorithm>
#include <map>
#include <filesystem>
#include <regex>

struct fuse_file_info;

namespace vfs{
    typedef int (*fuse_fill_dir_t) (void *buf, const char *name, const struct stat *stbuf, off_t off);

    class Mount;
    class Directory;
    class Video;

    class Inode {
    public:
        //TODO Think I can get rid of these
        //enum InodeType {
        //    directory, file, link, socket //, error
        //};

        Inode(const std::string &name, const Directory&, mode_t = 0644);
        Inode(std::string name, const Mount &mount, const Directory &parent, mode_t mode = 0644);
        Inode(const Inode &) = delete;
        Inode(Inode &&) = default;
        virtual ~Inode() = default;

        Inode& operator=(const Inode&) = delete;
        Inode& operator=(Inode&&) = default;

        const std::string &name() const { return name_; }

        mode_t mode() const { return mode_; }
        //InodeType type() const { return type_; }
        //void type(InodeType type) { type_ = type; }
        const Directory &parent() const { return parent_; }
        //void parent(Directory &parent) { parent_ = parent; }
        const Mount &mount() const { return mount_; }
        //void mount(Mount &mount) { mount_ = mount; }
        const std::filesystem::path &path() const { return path_; }

    //    virtual Inode * leaf(Path *path) { return this; }

        virtual int getattr(struct stat*) = 0;
        virtual int unlink() { return 0; }

    protected:
        const std::string name_;
        const std::filesystem::path path_;
        const Mount &mount_; //TODO remove
        const Directory &parent_;
        //const InodeType type_;
        const mode_t mode_;
    };

    /*class Error : public Inode {

        int error() const { return error_; }
        int getattr(struct stat*) override { return error(); }

    private:
        friend class Mount;

        explicit Error(const Mount& mount, int error);

        const int error_;
    };*/

    class Directory : public Inode {
    public:
        Directory(const std::string &name, const Directory &parent, mode_t mode)
                : Inode(name, parent, mode), files_(0), directories_(0)
        { }
        Directory(const Mount &mount, mode_t mode)
                : Inode({}, mount, *this, mode), files_(0), directories_(0)
        { }

        virtual std::optional<std::reference_wrapper<Inode>> find_child(const std::string &name) {
            auto it = std::find_if(children_.begin(), children_.end(),
                                   [&name](const auto &c) { return c->name() == name; });
            return it != children_.end()
                ? std::optional<std::reference_wrapper<Inode>>{**it}
                : std::nullopt;
        }

        virtual Inode& emplace_child(std::unique_ptr<Inode> &&child) {
            children_.emplace_back(std::move(child));
            return *children_.back();
        }

        template<typename T>
        T& emplace_child(std::unique_ptr<T> &&child) {
            children_.emplace_back(std::move(child));
            return dynamic_cast<T&>(*children_.back());
        }

        //Inode * leaf(Path *path) override;
        //void add_child(const std::string &name, std::unique_ptr<Inode> node);
        //void remove_child(const std::string &name);
        int getattr(struct stat*) override;
        int readdir(void *buffer, vfs::fuse_fill_dir_t, off_t, struct fuse_file_info&);
        //virtual int mkdir(const char *name, mode_t mode) { return -EACCES; }
        //virtual int mknod(const char *name, mode_t mode, dev_t rdev);
        //virtual int create(const char *name, mode_t mode, struct fuse_file_info *fi) { return -ENOTSUP; }
        //virtual int unlink(const char *name);
        //std::string path(const Inode *node) const;
    protected:
        std::vector<std::unique_ptr<Inode>> &children() {
            return children_;
            //std::vector<std::reference_wrapper<Inode>> values;
            //std::transform(children_.begin(), children_.end(), std::back_inserter(values),
            //               [](auto &pair) { return std::ref(*pair.second); });
            //return values;
        }

    private:
        std::vector<std::unique_ptr<Inode>> children_;
        size_t files_;
        size_t directories_;
    };

    class RootDirectory : public Directory {
    public:
        explicit RootDirectory(const Mount& mount, mode_t mode = 0644)
            : Directory(mount, mode)
        { }
        int mkdir(const char *name, mode_t mode);
    };

    class File : public Inode {
    public:
        File(const std::string &name, const Directory &directory, const mode_t mode)
                : Inode(name, directory, mode), size_(0)
        { }
        //virtual ~File() = default;
        int getattr(struct stat *st) override;
        virtual int open(struct fuse_file_info&);
        virtual int read(const std::filesystem::path&, char*, size_t, off_t, struct fuse_file_info&) = 0;
        virtual int write(const char*, size_t, off_t, struct fuse_file_info&) = 0;
        virtual int truncate(off_t) = 0; //{ return -EACCES; }
        virtual int flush(struct fuse_file_info&) { return 0; }
    protected:
        //virtual size_t size() const { return size_; } //= 0;
        //int read_helper(const std::string &data, char *buf, size_t size,
        //                off_t offset, struct fuse_file_info *fi);
    private:
        size_t size_;
    };

    class Link : public Inode {
    public:
        Link(const std::string &name, std::filesystem::path path, const Directory &directory, mode_t mode)
                : Inode(name, directory, mode), path_(std::move(path))
        { }

        int getattr(struct stat *st) override;
        virtual int readlink(char *buf, size_t size);

    private:
        std::filesystem::path path_;
    };

} // namespace vfs

#endif //VFS_INODE_H
