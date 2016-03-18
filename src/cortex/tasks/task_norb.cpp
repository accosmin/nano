#include "archive.h"
#include "task_norb.h"
#include "io/archive.h"
#include "io/imstream.h"
#include "cortex/class.h"
#include "cortex/logger.h"

namespace nano
{
        static const strings_t tlabels =
        {
                "animal",
                "human",
                "plane",
                "truck",
                "car",
                "blank"
        };

        norb_task_t::norb_task_t(const string_t&) :
                mem_vision_task_t("norb", 1, 108, 108, 5)
        {
        }

        bool norb_task_t::populate(const string_t& dir)
        {
                const size_t train_size = 29160;// * 10;
                const size_t test_size = 29160;// * 2;

                clear_memory(train_size + test_size);

                return  load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-01", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-02", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-03", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-04", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-05", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-06", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-07", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-08", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-09", protocol::train, train_size) &&
                        load_binary(dir + "/norb-5x46789x9x18x6x2x108x108-training-10", protocol::train, train_size) &&

                        load_binary(dir + "/norb-5x01235x9x18x6x2x108x108-testing-01", protocol::test, test_size) &&
                        load_binary(dir + "/norb-5x01235x9x18x6x2x108x108-testing-02", protocol::test, test_size);
        }

        static bool read_header(nano::imstream_t& stream, int32_t& magic, std::vector<int32_t>& dims)
        {
                // read data type & #dimensions
                int32_t ndims;
                if (    !stream.read((char*)&magic, sizeof(int32_t)) ||
                        !stream.read((char*)&ndims, sizeof(int32_t)))
                {
                        return false;
                }

                // read 3D/4D dimensions
                int32_t dim1, dim2, dim3, dim4;
                if (    !stream.read((char*)&dim1, sizeof(int32_t)) ||
                        !stream.read((char*)&dim2, sizeof(int32_t)) ||
                        !stream.read((char*)&dim3, sizeof(int32_t)) ||
                        (ndims > 3 && !stream.read((char*)&dim4, sizeof(int32_t))))
                {
                        return false;
                }

                // OK
                dims.push_back(dim1);
                dims.push_back(dim2);
                dims.push_back(dim3);
                if (ndims > 3)
                {
                        dims.push_back(dim4);
                }

                return true;
        }

        bool norb_task_t::load_binary(const string_t& bfile, protocol p, size_t count)
        {
                return load_binary(bfile + "-dat.mat.gz", bfile + "-cat.mat.gz", p, count);
        }

        bool norb_task_t::load_binary(const string_t& ifile, const string_t& gfile, protocol p, size_t count)
        {
//                 static const int magic_f32 = 0x1E3D4C51;
//                 static const int magic_f64 = 0x1E3D4C53;
                static const int magic_i32 = 0x1E3D4C54;
                static const int magic_i08 = 0x1E3D4C55;
//                 static const int magic_i16 = 0x1E3D4C56;

                size_t iindex = n_images();
                size_t icount = 0;
                size_t gcount = 0;

                const auto error_op = [&] (const string_t& message)
                {
                        log_error() << "NORB: " << message;
                };

                // load images
                const auto iop = [&] (const string_t&, const nano::buffer_t& data)
                {
                        nano::imstream_t stream(data.data(), data.size());

                        // read header
                        int32_t magic;
                        std::vector<int32_t> dims;
                        if (!read_header(stream, magic, dims))
                        {
                                log_error() << "NORB: failed to read header!";
                                return false;
                        }

                        if (    magic != magic_i08 ||

                                dims.size() != 4 ||
                                dims[1] != 2 ||
                                dims[2] != static_cast<int>(irows()) ||
                                dims[3] != static_cast<int>(icols()))
                        {
                                log_error() << "NORB: invalid header!";
                                return false;
                        }

                        // load images
                        const size_t n_cameras = 2;
                        const auto buffer_size = irows() * icols();
                        const auto cnt = dims[0];

                        nano::buffer_t buffer = nano::make_buffer(buffer_size);
                        for (auto i = 0; i < cnt; ++ i)
                        {
                                for (size_t cam = 0; cam < n_cameras && stream.read(buffer.data(), buffer_size); ++ cam)
                                {
                                        image_t image;
                                        image.load_luma(buffer.data(), irows(), icols());
                                        add_image(image);
                                }

                                ++ icount;
                        }

                        return stream.tellg() == stream.size();
                };

                log_info() << "NORB: loading file <" << ifile << "> ...";
                if (!nano::unarchive(ifile, iop, error_op))
                {
                        log_error() << "NORB: failed to load file <" << ifile << ">!";
                        return false;
                }

                // load ground truth
                const auto gop = [&] (const string_t& filename, const nano::buffer_t& data)
                {
                        NANO_UNUSED1(filename);

                        nano::imstream_t stream(data.data(), data.size());

                        // read header
                        int32_t magic;
                        std::vector<int32_t> dims;
                        if (!read_header(stream, magic, dims))
                        {
                                log_error() << "NORB: failed to read header!";
                                return false;
                        }

                        if (    magic != magic_i32 ||

                                dims.size() != 3 ||
                                dims[1] != 1)
                        {
                                log_error() << "NORB: invalid header!";
                                return false;
                        }

                        // load annotations
                        const size_t n_cameras = 2;
                        const auto cnt = dims[0];

                        int32_t label;
                        for (auto i = 0; i < cnt && stream.read(reinterpret_cast<char*>(&label), sizeof(label)); ++ i)
                        {
                                const tensor_index_t ilabel = label;
                                for (size_t cam = 0; cam < n_cameras; ++ cam)
                                {
                                        sample_t sample(iindex, sample_region(0, 0));
                                        if (ilabel < osize())
                                        {
                                                sample.m_label = tlabels[static_cast<size_t>(ilabel)];
                                                sample.m_target = nano::class_target(ilabel, osize());
                                        }
                                        sample.m_fold = { 0, p };
                                        add_sample(sample);

                                        ++ iindex;
                                }

                                ++ gcount;
                        }

                        return stream.tellg() == stream.size();
                };

                log_info() << "NORB: loading file <" << gfile << "> ...";
                if (!nano::unarchive(gfile, gop, error_op))
                {
                        log_error() << "NORB: failed to load file <" << gfile << ">!";
                        return false;
                }

                // OK
                log_info() << "NORB: loaded " << icount << "/" << gcount << " samples.";
                return (count == gcount) && (count == icount);
        }
}
