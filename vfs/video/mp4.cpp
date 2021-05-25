#include "mp4.h"
#include "errors.h"
extern "C" {
#include "libavformat/avformat.h"
#include "libavutil/dict.h"
}
#include "gpac/isomedia.h"
#include "gpac/media_tools.h"
#include <ext/stdio_filebuf.h>
#include <filesystem>
#include <coroutine>

/*#include "ISOBMFF.hpp"

class MdatBox: public ISOBMFF::Box
{
    public:

        MdatBox( void ): Box( "mdat" )
        {}

        void ReadData(Parser & parser, BinaryStream &stream)
        {
        start = reinterpret_cast<std::ifstream&>(stream._stream).tellg();
        }

        std::vector<std::pair<std::string, std::string>> GetDisplayableProperties(void) const
        {
            return {};
        }

    streampos start;
};*/

std::vector<char> FrameIterator::nal_prefix_{0, 0, 1};

//<std::vector<char>>
void FrameIterator::next_frame()
{
    //auto start_offset = stsz_offsets_[start_frame];
    //auto bytes_remaining = stsz_offsets_[end_frame] - start_position;

    //file.seekg(mdat_offset + start_offset);
    //frame_data.insert(frame_data.end(), nal_prefix_.begin(), nal_prefix_.end());

    if(bytes_remaining_ > 0)
    {
        uint32_t frame_data_size;
        file_.read(reinterpret_cast<char*>(frame_data_size), 4);

        value_.reserve(nal_prefix_.size() + frame_data_size);
        file_.read(value_.data() + nal_prefix_.size(), frame_data_size);
        //value_.insert(value_.begin() + nal_prefix_.size(), iterator_, iterator_ + frame_data_size);

        bytes_remaining_ -= frame_data_size + sizeof(uint32_t);
        //co_yield frame_data;
    } else
        value_.clear();

//    assert(bytes_remaining == 0);
}

std::ofstream open_output(std::string filename) { std::ofstream stream; stream.open(filename); return stream; }

void write_raw(const int input_descriptor) //, std::ostream&, unsigned int start_frame, unsigned int end_frame, const size_t mdat_offset, const std::vector<size_t>& stsz_offsets)
//void write(std::istream& in, std::ostream& out, const unsigned int start_frame, const unsigned int end_frame, const size_t mdat_offset, const std::vector<size_t>& stsz_offsets)
{
__gnu_cxx::stdio_filebuf<char> filebuf(input_descriptor, std::ios::in);
std::istream is(&filebuf);
//std::ifstream in(::_fdopen(input_descriptor, "r")); // 1

printf("foo\n");
}




static std::vector<int> get_samples(GF_ISOFile *file, const unsigned int track_index) {
    GF_Err result;
    const char *url, *urn;
    std::vector<int> offsets;

//    const auto stream_count = gf_isom_get_sample_description_count(file, track_index);

//    if(stream_count > 1)
  //      throw GpacRuntimeError("Error opening file", GF_IO_ERR);

    const auto sample_count = gf_isom_get_sample_count(file, track_index);

    //printf("sc %d\n", sample_count);
    auto position = 0u;
    for(auto sample_index = 1u; sample_index <= sample_count; sample_index++)
    {
        const auto *sample = gf_isom_get_sample_info(file, track_index, sample_index, nullptr, nullptr);
        //if(sample->IsRAP)
        offsets.push_back(position + sample->dataLength);
        position += sample->dataLength;
    }

     return offsets;

//    gf_isom_get_sample_info();

/*
    for(auto stream = 1u; stream < streams + 1; stream++) {
        if((result = gf_isom_get_data_reference(file, track_index, stream, &url, &urn)) != GF_OK)
            throw GpacRuntimeError(gf_error_to_string(result), result);
        else if(strict && url == nullptr)
            throw GpacRuntimeError("No data reference associated with stream", GF_NOT_SUPPORTED);
        else if(strict && urn != nullptr)
            throw GpacRuntimeError("Unexpected urn associated with stream", GF_NOT_SUPPORTED);
        else {
            unsigned int height, width;
            unsigned int samples, scale;
            unsigned long duration;
            int left, top;

            auto bitrate = get_average_bitrate(file, track_index, stream);

            if((samples = gf_isom_get_sample_count(file, track_index)) == 0)
                throw GpacRuntimeError("Unexpected error getting sample count", GF_NOT_FOUND);
            else if((duration = gf_isom_get_media_duration(file, track_index)) == 0)
                throw GpacRuntimeError("Unexpected error getting duration", GF_NOT_FOUND);
            else if((scale = gf_isom_get_media_timescale(file, track_index)) == 0)
                throw GpacRuntimeError("Unexpected error getting scale", GF_NOT_FOUND);
            else if((result = gf_isom_get_track_layout_info(file, track_index, &width, &height, &left, &top, nullptr)) != GF_OK)
                throw GpacRuntimeError("Unexpected error getting track layout", result);

            CHECK_GE(left, 0);
            CHECK_GE(top, 0);

            auto geometry = entry != nullptr
                            ? serialization::as_geometry(*entry)
                            : default_geometry.value();
            auto volume = entry != nullptr
                          ? serialization::as_composite_volume(*entry)
                          : default_volume.value() | TemporalRange{default_volume->t().start(),
                                                                   default_volume->t().end() + duration / scale};

            if(entry == nullptr && default_volume.has_value() && volume.bounding().t() != default_volume->t())
                LOG(INFO) << "Video duration did not match specified temporal range and was automatically increased.";

            sources.emplace_back(catalog::Source{
                    track_index - 1,
                    std::filesystem::absolute(url != nullptr ? url : filename),
                    get_codec(file, track_index, stream),
                    Configuration{
                            width,
                            height,
                            0u,
                            0u,
                            bitrate,
                            {scale * samples, static_cast<unsigned int>(duration)},
                            {static_cast<unsigned int>(left), static_cast<unsigned int>(top)}},
                    volume,
                    geometry});
        }
    }

    return sources;
*/
}

std::vector<int> unmux(const std::string &filename)
{
    GF_Err result;
    GF_ISOFile *file;

 AVFormatContext *fmt_ctx = NULL;
    AVDictionaryEntry *tag = NULL;
    int ret;

    av_register_all();

//    printf("%d\n", file->mdat);

    if ((ret = avformat_open_input(&fmt_ctx, filename.c_str(), NULL, NULL)))
        printf("ret %d\n", ret);
    while ((tag = av_dict_get(fmt_ctx->metadata, "", tag, AV_DICT_IGNORE_SUFFIX)))
        printf("%s=%s\n", tag->key, tag->value);
    //avformat_close_input(&fmt_ctx);

    if((file = gf_isom_open(filename.c_str(), GF_ISOM_OPEN_READ_DUMP, nullptr)) == nullptr)
        throw GpacRuntimeError("Error opening file", GF_IO_ERR);

    auto samples = get_samples(file, 1);
    //for(auto s: samples)
     //   printf("%d\n", s);
/*
    for(auto track = 1u; track < tracks + 1; track++)
        results.emplace_back(load_track(file, metadata.has_value()
                                            ? &metadata->entries(track - 1)
                                            : nullptr,
                                        filename, track, strict));
*/
    if((result = gf_isom_close(file)) != GF_OK)
        throw GpacRuntimeError("Error closing file", result);

    return samples;
}
