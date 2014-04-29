#include "task_svhn.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "common/io_zlib.h"
#include "common/io_mat5.h"
#include "loss.h"
#include "color.h"
#include <fstream>
#include <memory>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        bool svhn_task_t::load(const string_t& dir)
        {
                const string_t train_file = dir + "/train_32x32.mat";
                const string_t extra_file = dir + "/extra_32x32.mat";
                const size_t n_train_images = 73257 + 531131;

                const string_t test_file = dir + "/test_32x32.mat";
                const size_t n_test_images = 26032;

                m_images.clear();
                m_folds.clear();

                return  load(train_file, protocol::train) +
                        load(extra_file, protocol::train) == n_train_images &&
                        load(test_file, protocol::test) == n_test_images &&
                        build_folds(n_train_images, n_test_images);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t svhn_task_t::load(const string_t& bfile, protocol p)
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
                std::vector<u_int8_t> image_data;
                std::vector<u_int8_t> label_data;
                for (int isection = 0; isection < 2; isection ++)
                {
                        // section header
                        mat5::section_t section;
                        if (!section.load(istream))
                        {
                                log_error() << "SVHN: failed to read section!";
                                return 0;
                        }

                        if (section.m_dtype != mat5::data_type::miCOMPRESSED)
                        {
                                log_error() << "SVHN: invalid data type <" << mat5::to_string(section.m_dtype)
                                            << ">! expecting " << mat5::to_string(mat5::data_type::miCOMPRESSED) << "!";
                                return 0;
                        }

                        log_info() << "SVHN: uncompressing " << section.dsize() << " bytes ...";

                        std::vector<u_int8_t>& data = (isection == 0) ? image_data : label_data;
                        if (!io::zuncompress(istream, section.dsize(), data))
                        {
                                log_error() << "SVHN: failed to read compressed data!";
                                return 0;
                        }

                        log_info() << "SVHN: uncompressed " << data.size() << " bytes.";
                }

                // decode the uncompressed bytes
                return decode(image_data, label_data, p);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t svhn_task_t::decode(
                const std::vector<u_int8_t>& idata,
                const std::vector<u_int8_t>& ldata,
                protocol p)
        {
                // decode image & label arrays
                mat5::array_t iarray, larray;
                if (!iarray.load(idata))
                {
                        log_error() << "SVHN: invalid image array!";
                        return 0;
                }
                if (!larray.load(ldata))
                {
                        log_error() << "SVHN: invalid label array!";
                        return 0;
                }

                iarray.log(log_info() << "SVHN: image array: ");
                larray.log(log_info() << "SVHN: label array: ");

                const mat5::section_t& isection = iarray.m_sections[3];
                const mat5::section_t& lsection = larray.m_sections[3];

                const indices_t& idims = iarray.m_dims;
                const indices_t& ldims = larray.m_dims;

                // check array size
                if (    idims.size() != 4 ||
                        idims[0] != n_rows() ||
                        idims[1] != n_cols() ||
                        idims[2] != 3 ||

                        ldims.size() != 2 ||
                        ldims[1] != 1 ||

                        idims[3] != ldims[0])
                {
                        log_error() << "SVHN: invalid or mis-matching image & label array size!";
                        return 0;
                }

                // check data type
                if (    isection.m_dtype != mat5::data_type::miUINT8 ||
                        lsection.m_dtype != mat5::data_type::miUINT8)
                {
                        log_error() << "SVHN: expecting UINT8 image & label arrays!";
                        return 0;
                }

                // load images & labels
                const size_t n_samples = idims[3];

                size_t cnt = 0;
                for (size_t i = 0; i < n_samples; i ++)
                {
                        const size_t lbeg = lsection.dbegin() + i;

                        // label ...
                        size_t ilabel = ldata[lbeg + 0];
                        if (ilabel == 10)
                        {
                                ilabel = 0;
                        }
                        else if (ilabel < 1 || ilabel > 9)
                        {
                                continue;
                        }

                        const annotation_t anno(sample_region(0, 0),
                                "digit" + text::to_string(ilabel),
                                ncv::class_target(ilabel, n_outputs()));

                        // image ...
                        image_t image;
                        image.m_protocol = p;
                        image.m_annotations.push_back(anno);
                        image.m_rgba.resize(n_rows(), n_cols());

                        const size_t px = n_rows() * n_cols();
                        const size_t ix = n_rows() * n_cols() * 3;
                        const size_t ibeg = isection.dbegin() + i * ix;

                        for (size_t r = 0, p = 0; r < n_rows(); r ++)
                        {
                                for (size_t c = 0; c < n_cols(); c ++, p ++)
                                {
                                        const size_t ir = ibeg + (px * 0 + p);
                                        const size_t ig = ibeg + (px * 1 + p);
                                        const size_t ib = ibeg + (px * 2 + p);

                                        image.m_rgba(c, r) = color::make_rgba(idata[ir], idata[ig], idata[ib]);
                                }
                        }

                        m_images.push_back(image);
                        ++ cnt;
                }

                return cnt;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
