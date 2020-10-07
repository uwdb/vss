#ifndef VFS_VIDEO_H
#define VFS_VIDEO_H

#include "nativevideo.h"
#include "virtualvideo.h"
#include "writablevideo.h"
#include "policy.h"

namespace vfs {
    class NativeVideo;

    class Video : public Directory {
    public:
        Video(const std::string&, const Mount&, mode_t); //, std::vector<std::filesystem::path> sources)

        std::optional<std::reference_wrapper<Inode>> find_child(const std::string &name) override {
            auto child = Directory::find_child(name);

            return child.has_value()
                   ? child
                   : find_virtual_child(name);
        }

        template<typename T=Inode>
        T& substitute(const VirtualVideo &source, std::unique_ptr<T> &&target) {
            children().erase(std::remove_if(children().begin(), children().end(),
                                            [&source](const auto &child) { return child.get() == &source; }),
                             children().end());
            children().push_back(std::move(target));
            return dynamic_cast<T&>(*children().back());
        }

        /*int write(const VirtualVideo &view, struct fuse_file_info &info) {
            //TODO need to ensure no readers/writers
            children().clear();



            //std::remove_if(children().begin(), children().end(), [](const auto &child) { return *child == view; });
            this->emplace_child(std::make_unique<NativeVideo>(view.name(), *this, view.mode()));
            return 0;
        }*/

        const Policy &policy() const { return policy_; }

        bool has_native_child() {
            return std::any_of(children().begin(), children().end(),
                               [](const auto &child) { return dynamic_cast<const NativeVideo*>(child.get()) != nullptr; });
        }

        void opened(VirtualVideo &video) {
            opened_.insert(video);
        }

        void closed(VirtualVideo &video) {
            opened_.erase(video);
        }

        //Inode * leaf(Path *path) override;
        //void add_child(const std::string &name, std::unique_ptr<Inode> node);
        //void remove_child(const std::string &name);
        //int getattr(struct stat *st) override;
        //int readdir(void *buffer, vfs::fuse_fill_dir_t, off_t, struct fuse_file_info*);
        //virtual int mkdir(const char *name, mode_t mode) { return -EACCES; }
        //virtual int mknod(const char *name, mode_t mode, dev_t rdev);
        //virtual int create(const char *name, mode_t mode, struct fuse_file_info *fi) { return -ENOTSUP; }
        //virtual int unlink(const char *name);
        //std::string path(const Inode *node) const;
    private:
        std::optional<std::reference_wrapper<Inode>> find_virtual_child(const std::string &name) {
            std::regex expression{R"((\d+)x(\d+)(?:c(\d+)-(\d+)-(\d+)-(\d+))?(?:t(\d+)-(\d+))?\.(hevc|h264|rgb|yuv))"};
            std::smatch matches;

            //if(std::regex_match(name, expression)) {
            if(std::regex_search(name, matches, expression)) {
                auto width = std::stoul(matches[1]);
                auto height = std::stoul(matches[2]);
                auto crop_left = std::stof('0' + matches[3].str()) / 100;
                auto crop_top = std::stof('0' + matches[4].str()) / 100;
                auto crop_width = std::stof('0' + matches[5].str()) / 100;
                auto crop_height = std::stof('0' + matches[6].str()) / 100;
                auto time_start = std::stof('0' + matches[7].str());
                auto time_end = std::stof('0' + matches[8].str());
                auto format = VideoFormat::get_from_extension(path() / name);
                auto framerate = 30u;
                auto gop_size = 300u;

                //if(!has_native_child() &&
                //   crop_left == 0 && crop_top == 0 &&
                //   crop_width == 0 && crop_height == 0 &&
                //   time_start == 0 && time_end == 0)
                //    return std::optional<std::reference_wrapper<Inode>>{std::reference_wrapper(emplace_child(std::make_unique<WritableVirtualVideo>(name, *this, width, height, 0777)))};
                //else if(has_native_child())
                    return std::optional<std::reference_wrapper<Inode>>{std::reference_wrapper(emplace_child(std::make_unique<VirtualVideo>(name, *this, format, height, width, framerate, gop_size, 0777)))};
                //else
                //    return {};
            }
            //auto it = match.begin();

            //std::string x = it++;

            //  printf("Match\n");
            //return std::optional<std::reference_wrapper<Inode>>{std::reference_wrapper(emplace_child(std::make_unique<File>(name, *this, 0777)))};
            //}

            return {};
        }

        Policy policy_;
        std::set<std::reference_wrapper<VirtualVideo>,
                 std::function<bool(const std::reference_wrapper<VirtualVideo>&,
                                    const std::reference_wrapper<VirtualVideo>&)>> opened_;
    };

} // namespace vfs

#endif //VFS_VIDEO_H
