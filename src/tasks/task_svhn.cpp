#include "class.h"
#include "logger.h"
#include "io/mat5.h"
#include "task_svhn.h"
#include "math/random.h"
#include "vision/color.h"
#include "text/to_params.h"
#include "text/concatenate.h"
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

        size_t svhn_task_t::load_binary(const string_t& path, const protocol p)
        {
                log_info() << "SVHN: processing file <" << path << "> ...";

                const auto ecallback = [&] (const string_t& message)
                {
                        log_error() << "SVHN: " << message;
                };

                // load header
                const auto hcallback = [&] (const mat5_header_t& header)
                {
                        log_info() << "SVHN: header <" << header.description() << ">.";
                        return true;
                };

                // load section
                string_t name;
                std::vector<int32_t> dims;
                size_t icount = 0;
                size_t lcount = 0;
                size_t chunk_index = n_chunks();

                int section_index = 0;
                const auto scallback = [&] (const mat5_section_t& section, istream_t& stream)
                {
                        log_info() << "SVHN: section <" << section << ">.";
                        switch (section_index ++)
                        {
                        // pixels matrix section: {meta, dimensions, name, data} sub-elements
                        case 0:         return section.matrix_meta(stream);
                        case 1:         return section.matrix_dims(stream, dims);
                        case 2:         return section.matrix_name(stream, name);
                        case 3:         return section.matrix_data(stream) &&
                                               (icount = load_pixels(section, name, dims, stream)) > 0;

                        // labels matrix section: {meta, dimensions, name, data} sub-elements
                        case 4:         return section.matrix_meta(stream);
                        case 5:         return section.matrix_dims(stream, dims);
                        case 6:         return section.matrix_name(stream, name);
                        case 7:         return section.matrix_data(stream) &&
                                               (lcount = load_labels(section, name, dims, chunk_index, p, stream)) > 0;

                        default:        log_error() << "SVHN: unexpected section!"; return false;
                        }
                };

                return  (load_mat5(path, hcallback, scallback, ecallback) && icount == lcount) ?
                        icount : size_t(0);
        }

        size_t svhn_task_t::load_pixels(const mat5_section_t& section,
                const string_t& name, const std::vector<int32_t>& dims,
                istream_t& stream)
        {
                log_info() << "SVHN: loading images: name = " << name << ", size = " << concatenate(dims, "x") << "...";

                const auto px = irows() * icols();
                const auto ix = irows() * icols() * 3;

                // check input
                if (    dims.size() != 4 ||
                        static_cast<tensor_size_t>(dims[0]) != irows() ||
                        static_cast<tensor_size_t>(dims[1]) != icols() ||
                        dims[2] != 3 ||
                        dims[3] * ix > section.m_dsize)
                {
                        log_error() << "SVHN: invalid image array size!";
                        return 0;
                }

                if (    section.m_dtype != mat5_data_type::miUINT8)
                {
                        log_error() << "SVHN: expecting miUINT8 image array!";
                        return 0;
                }

                const auto n_samples = static_cast<size_t>(dims[3]);

                // load images
                auto idata = make_buffer(ix);
                for (size_t i = 0; i < n_samples; ++ i)
                {
                        if (stream.read(idata.data(), ix) != ix)
                        {
                                log_error() << "SVHN: failed to load image!";
                                return 0;
                        }

                        image_t image(irows(), icols(), color_mode::rgb);
                        image.plane(0) = tensor::map_matrix(idata.data() + 0 * px, icols(), irows()).cast<luma_t>().transpose();
                        image.plane(1) = tensor::map_matrix(idata.data() + 1 * px, icols(), irows()).cast<luma_t>().transpose();
                        image.plane(2) = tensor::map_matrix(idata.data() + 2 * px, icols(), irows()).cast<luma_t>().transpose();
                        add_chunk(image, image.hash());
                }

                return stream.skip(section.m_dsize - n_samples * ix) ? n_samples : 0;
        }

        size_t svhn_task_t::load_labels(const mat5_section_t& section,
                const string_t& name, const std::vector<int32_t>& dims, size_t chunk_index, const protocol p,
                istream_t& stream)
        {
                log_info() << "SVHN: loading labels: name = " << name << ", size = " << concatenate(dims, "x") << "...";

                // check input
                if (    dims.size() != 2 ||
                        dims[1] != 1 ||
                        dims[0] > section.m_dsize)
                {
                        log_error() << "SVHN: invalid label array size!";
                        return 0;
                }

                if (    section.m_dtype != nano::mat5_data_type::miUINT8)
                {
                        log_error() << "SVHN: expecting miUINT8 label array!";
                        return 0;
                }

                const auto n_samples = static_cast<size_t>(dims[0]);

                // load labels
                char ldata;
                for (size_t i = 0; i < n_samples; ++ i)
                {
                        if (stream.read(&ldata, 1) != 1)
                        {
                                log_error() << "SVHN: failed to load label!";
                                return 0;
                        }

                        tensor_size_t ilabel = ldata;
                        if (ilabel == 10)
                        {
                                ilabel = 0;
                        }
                        else if (ilabel < 1 || ilabel > 9)
                        {
                                log_error() << "SVHN: invalid label <" << ilabel << ">!";
                                return 0;
                        }

                        // target ...
                        const auto fold = make_fold(0, p);
                        add_sample(fold, chunk_index ++, class_target(ilabel, odims()), "digit" + to_string(ilabel));
                }

                return stream.skip(section.m_dsize - n_samples) ? n_samples : 0;
       }
}
