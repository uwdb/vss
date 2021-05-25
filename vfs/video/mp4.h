#ifndef VFS_MP4_H
#define VFS_MP4_H

#include <fstream>
#include <ostream>
#include <vector>
#include <string>

std::vector<int> unmux(const std::string &filename);
std::ofstream open_output(std::string filename); // { std::ifstream stream; stream.open(filename); return stream; }
void write_raw(const int); //, std::ostream&, unsigned int start_frame, unsigned int end_frame, const size_t mdat_offset, const std::vector<size_t>& stsz_offsets);
//void write(/*std::istream& in, std::ostream& out,*/ unsigned int start_frame, unsigned int end_frame, const unsigned int mdat_offset /*, const std::vector<size_t>& stsz_offsets*/);

class FrameIterator
{
public:
FrameIterator(std::istream& file, const unsigned int start_frame, const unsigned int end_frame, const size_t mdat_offset, const std::vector<size_t>& stsz_offsets)
    : file_(file), bytes_remaining_(stsz_offsets[end_frame] - stsz_offsets[start_frame]), value_(nal_prefix_)
    //start_frame_(start_frame), end_frame_(end_frame), mdat_offset_(mdat_offset), stsz_offsets_(stsz_offsets)
{
    auto start_offset = stsz_offsets[start_frame];

    file.seekg(mdat_offset + start_offset);
    //value.insert(frame_data.end(), nal_prefix_.begin(), nal_prefix_.end());
}

const std::vector<char>& operator*() const { return value_; }
FrameIterator& operator++() { next_frame(); return *this; }
FrameIterator operator++(int) { FrameIterator current = *this; ++(*this); return current; }

private:
void next_frame();

static std::vector<char> nal_prefix_; //{0, 0, 1};
size_t bytes_remaining_;
std::vector<char> value_;
std::istream& file_;
//const size_t start_frame_, end_frame_, mdat_offset_;
//const std::vector<size_t>& stsz_offsets;
};

#endif //VFS_MP4_H
