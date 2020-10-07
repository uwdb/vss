#ifndef VFS_COMPRESSIONWRITER_H
#define VFS_COMPRESSIONWRITER_H

#include "VideoEncoderSession.h"
#include "jointcompression.h"
#include "videowriter.h"
#include "homography.h"

namespace vfs {

    class CompressionWriter: public VideoWriter {
    public:
        explicit CompressionWriter(std::filesystem::path path, const lightdb::Codec &codec,
                                   const unsigned int framerate, const size_t gop_size)
            : path_{std::move(path)},
              context_{0},
              lock_{context_},
              configuration_{Configuration{0, 0, 0, 0, 4*1024*1024, {framerate, 1}, {0, 0}}, codec.nvidiaId().value(), 0},
              gop_{0u},
              gop_size_{gop_size}
        { }

        //~CompressionWriter() {
            //cuMemFree((CUdeviceptr)unpitched);
        //}

        void write(const graphics::GpuImage<3> &frame, const NppiRect &region={0,0,0,0}) {
            if(encoder_ == nullptr)
                initialize_encoder(frame.height(), frame.width());

            if(session_ == nullptr || session_->frameCount() == gop_size_)
                initialize_session();

            auto width = region.width ? region.width: frame.swidth();
            auto height = region.height ? region.height : frame.sheight();

            assert(frame.width() == gpu_frame_->width() &&
                   frame.height() == gpu_frame_->height());

            auto result = nppiRGBToYUV420_8u_C3P3R(
                    reinterpret_cast<const Npp8u*>(frame.device()),
                    frame.step(),
                    plane_offsets_.data(),
                    plane_pitches_.data(),
                    {width, height});
                    //{frame.swidth(), frame.sheight()});
            assert(result == NPP_SUCCESS);

/*                std::vector<unsigned char> foo;
                foo.resize(frame.height() * frame.width() * 3/2);
                auto r = cudaMemcpy2D(foo.data(), frame.width(), unpitched, frame.width(), frame.width(), frame.height() * 3/2, cudaMemcpyDeviceToHost);
                //auto r = cudaMemcpy2D(foo.data(), frame.width(), (void*)gpu_frame_->handle(), gpu_frame_->pitch(), frame.width(), frame.height() * 3/2, cudaMemcpyDeviceToHost);
                assert(r == cudaSuccess);*/

//            auto rx = cudaMemcpy2D((void*)gpu_frame_->handle(), gpu_frame_->pitch(), (void*)unpitched, frame.width(), frame.width(), frame.height() * 3/2, cudaMemcpyDeviceToDevice);
            //auto rx = cudaMemcpy2D((void*)gpu_frame_->handle(), gpu_frame_->pitch(), (void*)foo.data(), frame.width(), frame.width(), frame.height() * 3/2, cudaMemcpyHostToDevice);
            //auto rx = cudaMemcpy((void*)gpu_frame_->handle(), (void*)foo.data(), frame.width() * frame.height() * 3/2, cudaMemcpyHostToDevice);
  //          assert(rx == cudaSuccess);

            session_->Encode(*gpu_frame_);
        }

        void flush() /* override*/ {
            if(session_ != nullptr)
                session_->Flush();
        }

        void write(const std::vector<unsigned char>::iterator &left,
                   const std::vector<unsigned char>::iterator &right) override {
            if(frame_ == nullptr)
                //TODO
                frame_ = std::make_unique<graphics::GpuImage<3, Npp8u>>(nppiMalloc_8u_C3, 720, 1280);

            frame_->upload(left);
            write(*frame_);
            frame_->upload(right);
            write(*frame_);
        }

    private:
        //unsigned char *unpitched;
        void initialize_encoder(const size_t height, const size_t width) {
            configuration_ = {Configuration{static_cast<unsigned int>(width), static_cast<unsigned int>(height), 0, 0, configuration_.bitrate, configuration_.framerate, {0, 0}}, configuration_.codec, static_cast<unsigned int>(gop_size_)};
            encoder_ = std::make_unique<VideoEncoder>(context_, configuration_, lock_, NV_ENC_BUFFER_FORMAT_YV12);
            //cudaMalloc(&unpitched, height * width * 3/2);
            gpu_frame_ = std::make_unique<CudaFrame>(configuration_.height, configuration_.width, NV_ENC_PIC_STRUCT_FRAME, NV_ENC_BUFFER_FORMAT_YV12);
            //gpu_frame_ = std::make_unique<CudaFrame>(configuration_.height, configuration_.width, NV_ENC_PIC_STRUCT_FRAME, (CUdeviceptr)handle, configuration_.width);
            //temp = std::make_unique<CudaFrame>(configuration_.height, configuration_.width, NV_ENC_PIC_STRUCT_FRAME, NV_ENC_BUFFER_FORMAT_YV12);
            //plane_offsets_ = {
            //        /* y */ reinterpret_cast<Npp8u*>(unpitched),
            //        /* v */ reinterpret_cast<Npp8u*>(unpitched + (height * width) + (height * width) / 4),
            //        /* u */ reinterpret_cast<Npp8u*>(unpitched + height * width)};
            plane_offsets_ = {
                    /* y */ reinterpret_cast<Npp8u*>(gpu_frame_->handle()),
                    /* v */ reinterpret_cast<Npp8u*>(gpu_frame_->handle() + (5 * configuration_.height / 4) * gpu_frame_->pitch()),
                    /* u */ reinterpret_cast<Npp8u*>(gpu_frame_->handle() + configuration_.height * gpu_frame_->pitch())};
            //plane_pitches_ = {
            //        /* y */ static_cast<int>(width), //gpu_frame_->pitch()),
            //        /* v */ static_cast<int>(width / 2), //gpu_frame_->pitch() / 2),
            //        /* u */ static_cast<int>(width / 2) }; //gpu_frame_->pitch() / 2) };
            plane_pitches_ = {
                    /* y */ static_cast<int>(gpu_frame_->pitch()),
                    /* v */ static_cast<int>(gpu_frame_->pitch() / 2),
                    /* u */ static_cast<int>(gpu_frame_->pitch() / 2) };
            //cudaMemset2D((void*)gpu_frame_->handle(), gpu_frame_->pitch(), 55, width, height);
        }

        void initialize_session() {
            gop_++;
            session_ = nullptr;
            writer_ = std::make_unique<FileEncodeWriter>(*encoder_, path_ / (std::to_string(gop_ - 1) + ".h264"));
            session_ = std::make_unique<VideoEncoderSession>(*encoder_, *writer_);
        }

        //std::unique_ptr<CudaFrame> temp;

        const std::filesystem::path path_;
        GPUContext context_;
        VideoLock lock_;
        EncodeConfiguration configuration_;
        std::unique_ptr<VideoEncoder> encoder_;
        std::unique_ptr<CudaFrame> gpu_frame_;
        std::array<Npp8u*, 3> plane_offsets_;
        std::array<int, 3> plane_pitches_;
        size_t gop_;
        std::unique_ptr<FileEncodeWriter> writer_;
        std::unique_ptr<VideoEncoderSession> session_;
        const size_t gop_size_;
        std::unique_ptr<graphics::GpuImage<3, Npp8u>> frame_;
    };

} // namespace vfs
#endif //VFS_COMPRESSIONWRITER_H
