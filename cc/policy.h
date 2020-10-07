#ifndef VFS_POLICY_H
#define VFS_POLICY_H

#include <set>
#include "Codec.h"
#include "video.h"

namespace vfs {

class Policy {
public:
    explicit Policy(const Video&)
        : joint_{"/wolf"},
          default_codec_(lightdb::Codec::h264())
    {}

    const std::set<std::filesystem::path>& joint() const { return joint_; }
    const lightdb::Codec &default_codec() const { return default_codec_; }

private:
    std::set<std::filesystem::path> joint_;
    const lightdb::Codec default_codec_;
};
}

#endif //VFS_POLICY_H
