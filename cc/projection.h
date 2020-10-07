#ifndef VFS_PROJECTION_H
#define VFS_PROJECTION_H

#include "homography.h"

namespace vfs::graphics {
    class PartitionBuffer {
    public:
        PartitionBuffer(const GpuImage<3, Npp8u> &input, const Homography &homography)
            : homography_(homography),
              partitions_(homography.partitions(input.size())),
              //widths_{input.width()partitions_.left.x0,
              widths_{input.swidth() - partitions_.left.x1, // + partitions_.left.x1 % 2,
                      partitions_.left.x1, // - partitions_.left.x1 % 2,
                      partitions_.right.x1 /*robocar- partitions_.left.x1*/}, // + partitions_.right.x1 % 2},
              //widths_{partitions_.left.x1 + partitions_.left.x1 % 2,
              //        input.width() - partitions_.left.x1 - partitions_.left.x1 % 2,
              //        //partitions_.left.x1 - partitions_.left.x0,
              //        input.width() - partitions_.right.x1 + partitions_.right.x1 % 2},
              //widths_{partitions_.left.x0,
              //        partitions_.left.x1 - partitions_.left.x0,
              //        input.width() - partitions_.right.x1},
              right_height_{2*partitions_.right.y1},
              frames_{make_frame(input.allocator(), input.height(), widths_.left),
                      make_frame(input.allocator(), input.height() + right_height_, widths_.overlap), //input.width()),
                      //make_frame(input.allocator(), input.height(), widths_.overlap),
                      make_frame(input.allocator(), input.height(), widths_.right)}
        { }

        const Homography& homography() const { return homography_; }
        const Partitions& partitions() const { return partitions_; }
        const auto &widths() const { return widths_; }

        bool has_left() const { return frames_.left != nullptr; }
        bool has_overlap() const { return frames_.overlap != nullptr; }
        bool has_right() const { return frames_.right != nullptr; }
        GpuImage<3, Npp8u>& left() const { return *frames_.left; }
        GpuImage<3, Npp8u>& overlap() const { return *frames_.overlap; }
        GpuImage<3, Npp8u>& right() const { return *frames_.right; }

        struct Widths {
            ssize_t left, overlap, right;
        };

    private:
        std::unique_ptr<GpuImage<3, Npp8u>> make_frame(const GpuImage<3, Npp8u>::allocator_t &allocator,
                                                       const size_t height, const size_t width) {
            return height != 0 && width != 0
                ? std::make_unique<GpuImage<3, Npp8u>>(allocator, std::max(128lu, height), std::max(128lu, width))
                : nullptr;
        }

        Homography homography_;
        Partitions partitions_;
        Widths widths_;
        ssize_t right_height_;
        struct {
            std::unique_ptr<GpuImage<3, Npp8u>> left;
            std::unique_ptr<GpuImage<3, Npp8u>> overlap;
            std::unique_ptr<GpuImage<3, Npp8u>> right;
        } frames_;
    };

    PartitionBuffer& partition(const GpuImage<3, Npp8u> &left, const GpuImage<3, Npp8u> &right, PartitionBuffer &output);
    //std::tuple<GpuImage<3, Npp8u>, GpuImage<3, Npp8u>, GpuImage<3, Npp8u>> partition(
    //        const GpuImage<3, Npp8u>&, const Homography&);
    void project(const GpuImage<3, Npp8u> &input, GpuImage<3, Npp8u> &output, const Homography&, const NppiSize&);
}

#endif //VFS_PROJECTION_H
