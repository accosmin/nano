#include "class.h"
#include "logger.h"
#include "io/mat5.h"
#include "task_svhn.h"
#include "math/random.h"
#include "vision/color.h"
#include "text/to_params.h"
#include "io/istream_zlib.h"
#include "text/from_params.h"
#include <fstream>

namespace nano
{
        svhn_task_t::svhn_task_t(const string_t& config) :
                mem_vision_task_t(3, 32, 32, 10, 1, 1, 1, to_params(config, "dir", "."))
        {
        }

        bool svhn_task_t::populate()
        {
                const string_t dir = nano::from_params<string_t>(config(), "dir");

                const string_t train_file = dir + "/train_32x32.mat";
                const string_t extra_file = dir + "/extra_32x32.mat";
                const size_t n_train_samples = 73257 + 531131;

                const string_t test_file = dir + "/test_32x32.mat";
                const size_t n_test_samples = 26032;

                reserve_chunks(n_train_samples + n_test_samples);

                return  load_binary(train_file, protocol::train) +
                        load_binary(extra_file, protocol::train) == n_train_samples &&
                        load_binary(test_file, protocol::test) == n_test_samples;
        }

        size_t svhn_task_t::load_binary(const string_t& bfile, const protocol p)
        {
                log_info() << "SVHN: processing file <" << bfile << "> ...";

                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
                if (!istream.is_open())
                {
                        log_error() << "SVHN: failed to open file!";
                        return 0;
                }

                // header section
                char header[116];
                if (!istream.read(header, 116))
                {
                        log_error() << "SVHN: failed to read header!";
                        return 0;
                }
                log_info() << "SVHN: read header <" << string_t(header, header + 116) << ">.";

                char byte[8];
                if (    !istream.read(byte, 8) ||       // offset
                        !istream.read(byte, 4))         // version + endian
                {
                        log_error() << "SVHN: failed to read offset & version!";
                        return 0;
                }

                // data sections (image rgb + labels)
                size_t image_index = n_chunks();
                for (int isection = 0; isection < 2; ++ isection)
                {
                        // section header
                        nano::mat5_section_t section;
                        if (!section.load(istream))
                        {
                                log_error() << "SVHN: failed to read section!";
                                return 0;
                        }

                        if (section.m_dtype != nano::mat5_buffer_type::miCOMPRESSED)
                        {
                                log_error() << "SVHN: invalid data type <" << to_string(section.m_dtype)
                                            << ">! expecting " << to_string(nano::mat5_buffer_type::miCOMPRESSED) << "!";
                                return 0;
                        }

                        // section data
                        log_info() << "SVHN: uncompressing " << section.dsize() << " bytes ...";

                        zlib_istream_t dstream(istream, section.dsize());
                        switch (isection)
                        {
                        case 0: load_images(dstream, p); break;
                        case 1: load_labels(dstream, image_index, p); break;
                        }
                }
        }

        size_t svhn_task_t::load_images(zlib_istream_t& stream, const protocol p)
        {
                // decode image array
                nano::mat5_array_t array;
                if (    !array.load_header(stream) ||
                        !array.load_body(stream))
                {
                        log_error() << "SVHN: invalid image array!";
                        return 0;
                }

                log_info() << "SVHN: image array: " << array;

                const auto& section = array.m_sections[3];
                const auto& dims = array.m_dims;

                // check array size
                if (    dims.size() != 4 ||
                        static_cast<tensor_size_t>(dims[0]) != irows() ||
                        static_cast<tensor_size_t>(dims[1]) != icols() ||
                        dims[2] != 3)
                {
                        log_error() << "SVHN: invalid image array size!";
                        return 0;
                }

                // check data type
                if (section.m_dtype != nano::mat5_buffer_type::miUINT8)
                {
                        log_error() << "SVHN: expecting UINT8 image array!";
                        return 0;
                }

                // load images
                const size_t n_samples = dims[3];

                const auto px = irows() * icols();
                buffer_t idata(static_cast<size_t>(irows() * icols() * 3));
                const auto* iptr = idata.data();

                size_t cnt = 0;
                for ( ; cnt < n_samples && stream.read(idata.data(), idata.size()); ++ cnt)
                {
                        image_t image(irows(), icols(), color_mode::rgb);
                        image.plane(0) = tensor::map_matrix(iptr + 0 * px, icols(), irows()).cast<luma_t>().transpose();
                        image.plane(1) = tensor::map_matrix(iptr + 1 * px, icols(), irows()).cast<luma_t>().transpose();
                        image.plane(2) = tensor::map_matrix(iptr + 2 * px, icols(), irows()).cast<luma_t>().transpose();
                        add_chunk(image, image.hash());
                }

                return cnt;
        }

        size_t svhn_task_t::load_labels(zlib_istream_t& stream, size_t image_index, const protocol p)
        {
                // decode label array
                nano::mat5_array_t array;
                if (    !array.load_header(stream) ||
                        !array.load_body(stream))
                {
                        log_error() << "SVHN: invalid label array!";
                        return 0;
                }

                log_info() << "SVHN: label array: " << array;

                const auto& section = array.m_sections[3];
                const auto& dims = array.m_dims;

                // check array size
                if (    dims.size() != 2 ||
                        dims[1] != 1)
                {
                        log_error() << "SVHN: invalid label array size!";
                        return 0;
                }

                // check data type
                if (section.m_dtype != nano::mat5_buffer_type::miUINT8)
                {
                        log_error() << "SVHN: expecting UINT8 label array!";
                        return 0;
                }

                // load labels
                const size_t n_samples = dims[3];

                uint8_t ldata;

                size_t cnt = 0;
                for ( ; cnt < n_samples && stream.read(&ldata, 1); ++ cnt, ++ image_index)
                {
                        if (image_index >= n_chunks())
                        {
                                log_error() << "SVHN: mis-matching number of images and labels!";
                                return 0;
                        }

                        tensor_size_t ilabel = ldata;
                        if (ilabel == 10)
                        {
                                ilabel = 0;
                        }
                        else if (ilabel < 1 || ilabel > 9)
                        {
                                return 0;
                        }

                        const auto fold = make_fold(0, p);
                        add_sample(fold, image_index, class_target(ilabel, odims()), "digit" + to_string(ilabel));
                }

                return cnt;
        }
}
