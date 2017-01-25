#include "class.h"
#include "logger.h"
#include "io/mat5.h"
#include "task_svhn.h"
#include "math/random.h"
#include "vision/color.h"
#include "text/to_params.h"
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
                const auto scallback = [&] (const mat5_section_t& section, istream_t& stream)
                {
                        log_info() << "SVHN: section <" << section << ">.";
                        return true;
                };

                if (!load_mat5(bfile, hcallback, scallback, ecallback))
                {
                        return 0;
                }

                return 0;

                /*
                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
                if (!istream.is_open())
                {
                        log_error() << "SVHN: failed to open file!";
                        return 0;
                }

                // header section
                mat5_header_t header;
                if (!header.load(istream))
                {
                        log_error() << "SVHN: failed to read header!";
                        return 0;
                }
                log_info() << "SVHN: read header <" << header.description() << ">.";

                // data sections (image rgb + labels)
                nano::buffer_t image_data;
                nano::buffer_t label_data;
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

                        auto& data = (isection == 0) ? image_data : label_data;
                        if (!nano::uncompress_gzip(istream, section.dsize(), data))
                        {
                                log_error() << "SVHN: invalid gzip archive!";
                                return 0;
                        }

                        log_info() << "SVHN: uncompressed " << data.size() << " bytes.";
                }

                // decode the uncompressed bytes
                return decode(image_data, label_data, p);
*/
        }

        size_t svhn_task_t::decode(const nano::buffer_t& idata, const nano::buffer_t& ldata, const protocol p)
        {
                return 0;
                /*
                nano::imstream_t istream(idata.data(), idata.size());
                nano::imstream_t lstream(ldata.data(), ldata.size());

                // decode image & label arrays
                nano::mat5_array_t iarray, larray;
                if (    !iarray.load_header(istream) ||
                        !iarray.load_body(istream))
                {
                        log_error() << "SVHN: invalid image array!";
                        return 0;
                }
                if (    !larray.load_header(lstream) ||
                        !larray.load_body(lstream))
                {
                        log_error() << "SVHN: invalid label array!";
                        return 0;
                }

                log_info() << "SVHN: image array: " << iarray;
                log_info() << "SVHN: label array: " << larray;

                const auto& isection = iarray.m_sections[3];
                const auto& lsection = larray.m_sections[3];

                const auto& idims = iarray.m_dims;
                const auto& ldims = larray.m_dims;

                // check array size
                if (    idims.size() != 4 ||
                        static_cast<tensor_size_t>(idims[0]) != irows() ||
                        static_cast<tensor_size_t>(idims[1]) != icols() ||
                        idims[2] != 3 ||

                        ldims.size() != 2 ||
                        ldims[1] != 1 ||

                        idims[3] != ldims[0])
                {
                        log_error() << "SVHN: invalid or mis-matching image & label array size!";
                        return 0;
                }

                // check data type
                if (    isection.m_dtype != nano::mat5_buffer_type::miUINT8 ||
                        lsection.m_dtype != nano::mat5_buffer_type::miUINT8)
                {
                        log_error() << "SVHN: expecting UINT8 image & label arrays!";
                        return 0;
                }

                // load images & labels
                const size_t n_samples = idims[3];

                const auto px = irows() * icols();
                const auto ix = static_cast<size_t>(irows() * icols() * 3);
                auto iptr = idata.data();

                size_t cnt = 0;
                for (size_t i = 0; i < n_samples; ++ i, iptr += ix)
                {
                        const auto lbeg = static_cast<size_t>(lsection.dbegin()) + i;

                        // label ...
                        tensor_size_t ilabel = ldata[lbeg + 0];
                        if (ilabel == 10)
                        {
                                ilabel = 0;
                        }
                        else if (ilabel < 1 || ilabel > 9)
                        {
                                continue;
                        }

                        // image ...
                        image_t image(irows(), icols(), color_mode::rgb);
                        image.plane(0) = tensor::map_matrix(iptr + 0 * px, icols(), irows()).cast<luma_t>().transpose();
                        image.plane(1) = tensor::map_matrix(iptr + 1 * px, icols(), irows()).cast<luma_t>().transpose();
                        image.plane(2) = tensor::map_matrix(iptr + 2 * px, icols(), irows()).cast<luma_t>().transpose();
                        add_chunk(image, image.hash());

                        // target ...
                        const auto fold = make_fold(0, p);
                        add_sample(fold, n_chunks() - 1, class_target(ilabel, odims()), "digit" + to_string(ilabel));

                        ++ cnt;
                }

                return cnt;
                */
        }
}
