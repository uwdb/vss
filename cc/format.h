#ifndef VFS_FORMAT_H
#define VFS_FORMAT_H

#include "Codec.h"

class VideoFormat {
public:
    enum Value {
        RGB8,
        YUV422,
        H264,
        HEVC
    };

    static VideoFormat get_from_extension(const std::filesystem::path &path) {
        if(path.extension() == ".rgb")
            return RGB8;
        else if(path.extension() == ".yuv")
            return YUV422;
        else if(path.extension() == ".h264")
            return H264;
        else if(path.extension() == ".hevc")
            return HEVC;
        else
            throw std::runtime_error("Unrecognized extension");
    }

    constexpr size_t buffer_size(const size_t height, const size_t width) const {
        switch(value_) {
            case RGB8:
                return 16u * frame_size(height, width).value();
            case YUV422:
                return 16u * frame_size(height, width).value();
            case HEVC:
            case H264:
                return 128u * 1024u;
            default:
                throw std::runtime_error("Unsupported format");
        }
    }

    const std::optional<lightdb::Codec> codec() const {
        switch(value_) {
            case RGB8:
            case YUV422:
                return {};
            case HEVC:
                return lightdb::Codec::hevc();
            case H264:
                return lightdb::Codec::h264();
            default:
                throw std::runtime_error("Unsupported format");
        }
    }

    constexpr std::optional<size_t> frame_size(const size_t height, const size_t width) const {
        switch(value_) {
            case RGB8:
                return 3 * height * width;
            case YUV422:
                return 3/2 * height * width;
            case HEVC:
            case H264:
                return {};
            default:
                throw std::runtime_error("Unsupported format");
        }
    }

    VideoFormat() = default;
    constexpr VideoFormat(Value value) : value_(value) { }

    explicit operator bool() = delete;

    operator Value() const { return value_; }
private:
    Value value_;
};

#endif //VFS_FORMAT_H
