#include <vector>
#include <glog/logging.h>
#include "homography.h"

//#include <iostream>
//#include <cuda.h>
#include "/home/bhaynes/projects/CudaSift/cudaSift.h"

namespace vfs::graphics {
    namespace internal {
        struct SiftData: public ::SiftData {};
    }

    SiftConfiguration::SiftConfiguration(const size_t height, const size_t width,
                                         const float blur, const float threshold, const int octaves, const bool scale)
            : height_(height), width_(width), blur_(blur), threshold_(threshold), octaves_(octaves), scale_(scale),
              left_(std::make_unique<CudaImage>()), right_(std::make_unique<CudaImage>()),
              left_data_(std::make_unique<internal::SiftData>()), right_data_(std::make_unique<internal::SiftData>()),
              scratch_(AllocSiftTempMemory(static_cast<int>(width), static_cast<int>(height), octaves, scale)) {
        InitSiftData(*left_data_, 32768, false, true);
        InitSiftData(*right_data_, 32768, false, true);
    }

    SiftConfiguration::~SiftConfiguration() {
        FreeSiftTempMemory(scratch_);
        FreeSiftData(*left_data_);
        FreeSiftData(*right_data_);
    }

    Homography find_homography(const std::vector<unsigned char> &left, const std::vector<unsigned char> &right,
                               const size_t height, const size_t width) {
        SiftConfiguration configuration{height, width};
        return find_homography(left, right, height, width, configuration);
    }

    Homography find_homography(const std::vector<unsigned char> &left, const std::vector<unsigned char> &right,
                               const size_t height, const size_t width, SiftConfiguration &configuration) {
        return find_homography(GpuImage<3, Npp8u>{left, nppiMalloc_8u_C3, height, width},
                               GpuImage<3, Npp8u>{right, nppiMalloc_8u_C3, height, width}, configuration);
    }

    Homography find_homography(const GpuImage<3, Npp8u> &left, const GpuImage<3, Npp8u> &right,
                               SiftConfiguration &configuration) {
        GpuImage<1, Npp8u> left_gray{nppiMalloc_8u_C1, left.height(), left.width()};
        GpuImage<1, Npp8u> right_gray{nppiMalloc_8u_C1, right.height(), right.width()};

        if(nppiRGBToGray_8u_C3C1R(left.device(), left.step(), left_gray.device(), left_gray.step(), left.size()) != NPP_SUCCESS ||
           nppiRGBToGray_8u_C3C1R(right.device(), right.step(), right_gray.device(), right_gray.step(), right.size()) != NPP_SUCCESS)
            throw std::runtime_error("Failed to convert images to single-channel grayscale");

        return find_homography(left_gray, right_gray, configuration);
    }

    Homography find_homography(const GpuImage<1, Npp8u> &left, const GpuImage<1, Npp8u> &right,
                               SiftConfiguration &configuration) {
        GpuImage<1, Npp32f> left_float{nppiMalloc_32f_C1, left.height(), left.width()};
        GpuImage<1, Npp32f> right_float{nppiMalloc_32f_C1, right.height(), right.width()};

        if(nppiConvert_8u32f_C1R(left.device(), left.step(), left_float.device(), left_float.step(), left.size()) != NPP_SUCCESS ||
           nppiConvert_8u32f_C1R(right.device(), right.step(), right_float.device(), right_float.step(), right.size()) != NPP_SUCCESS)
            throw std::runtime_error("Failed to convert images to float");

        return find_homography(left_float, right_float, configuration);
    }

    Homography find_homography(const GpuImage<1, Npp32f> &left, const GpuImage<1, Npp32f> &right,
                               SiftConfiguration &configuration) {
        Homography homography{};
        int matches;

        configuration.left().Allocate(left.swidth(), left.sheight(), left.sbyte_step(), false, left.device(), nullptr);
        configuration.right().Allocate(right.swidth(), right.sheight(), right.sbyte_step(), false, right.device(), nullptr);

        ExtractSift(configuration.left_data(), configuration.left(), configuration.octaves(), configuration.blur(),
                    configuration.threshold(), 0.0f, false, configuration.scratch());
        ExtractSift(configuration.right_data(), configuration.right(), configuration.octaves(), configuration.blur(),
                    configuration.threshold(), 0.0f, false, configuration.scratch());

        MatchSiftData(configuration.left_data(), configuration.right_data());

        FindHomography(configuration.left_data(), static_cast<float*>(homography), &matches, 10000, 0.00f, 1/*0.80f*/, 4 /*5.0*/);

        LOG(INFO) << "SIFT: Number of original features: (" <<  configuration.left_data().numPts << ' ' <<
                                                                configuration.right_data().numPts << ')';

        return homography;
    }
}