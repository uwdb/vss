#ifndef VFS_JOINTCOMPRESSION_H
#define VFS_JOINTCOMPRESSION_H

#include "virtualvideo.h"
#include "homography.h"
#include "projection.h"
#include "compressionwriter.h"
#include "videowriter.h"

namespace vfs {
template<size_t channels>
class HomographyUpdateStrategy;

template<size_t channels>
class JointWriter: public VideoWriter {
public:
    template<typename HomographyUpdateStrategy>
    JointWriter(size_t height, size_t width, const HomographyUpdateStrategy &homography_update)
        : left_frame_(nppiMalloc_8u_C3, height, width),
          right_frame_(nppiMalloc_8u_C3, height, width),
          configuration_{height, width},
          partitions_{},
          homography_update_{std::make_unique<HomographyUpdateStrategy>(homography_update)},
          left_writer_{"/tmp/v/left", lightdb::Codec::h264(), 30, 300},
          overlap_writer_{"/tmp/v/overlap", lightdb::Codec::h264(), 30, 300},
          right_writer_{"/tmp/v/right", lightdb::Codec::h264(), 30, 300}
    { }

    void write(const std::vector<unsigned char>::iterator &left,
               const std::vector<unsigned char>::iterator &right) override {
        left_frame_.upload(left);
        right_frame_.upload(right);

        homography_update_->update(*this);
        assert(partitions_);

        graphics::project(right_frame_, partitions_->overlap(), partitions_->homography(),
                          {-partitions_->widths().left, (partitions_->overlap().sheight() - right_frame_.sheight()) / 2});
        graphics::partition(left_frame_, right_frame_, *partitions_);

        if(partitions_->has_left())
            left_writer_.write(partitions_->left());
        if(partitions_->has_overlap())
            overlap_writer_.write(partitions_->overlap());
        if(partitions_->has_right())
            right_writer_.write(partitions_->right());
    }

    void flush() override {
        if(partitions_ == nullptr)
            return;

        if(partitions_->has_left())
            left_writer_.flush();
        if(partitions_->has_overlap())
            overlap_writer_.flush();
        if(partitions_->has_right())
            right_writer_.flush();
    }

    const graphics::GpuImage<channels, Npp8u>& left_frame() const { return left_frame_; }
    const graphics::GpuImage<channels, Npp8u>& right_frame() const { return right_frame_; }

    bool has_homography() const { return partitions_ != nullptr; }
    void homography(const graphics::Homography & homography) {
        if(partitions_ == nullptr || homography != partitions_->homography())
            partitions_ = std::make_unique<graphics::PartitionBuffer>(left_frame_, homography);
    }
    graphics::SiftConfiguration& configuration() { return configuration_; }

private:
    graphics::GpuImage<channels, Npp8u> left_frame_;
    graphics::GpuImage<channels, Npp8u> right_frame_;

    graphics::SiftConfiguration configuration_;
    std::unique_ptr<graphics::PartitionBuffer> partitions_;
    std::unique_ptr<HomographyUpdateStrategy<channels>> homography_update_;

    //TODO this writer should be a composite
    CompressionWriter left_writer_;
    CompressionWriter overlap_writer_;
    CompressionWriter right_writer_;
};

template<size_t channels>
class HomographyUpdateStrategy {
public:
    virtual void update(JointWriter<channels>&) = 0;
};

class OneTimeHomographyUpdate : public HomographyUpdateStrategy<3> {
public:
    void update(JointWriter<3> &writer) override {
        if(!writer.has_homography())
            writer.homography(
                    find_homography(writer.left_frame(), writer.right_frame(),
                                    writer.configuration()));
    }
};

} // namespace vfs

#endif //VFS_JOINTCOMPRESSION_H
