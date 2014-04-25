#include "task_svhn.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "common/zcompress.h"
#include "loss.h"
#include "color.h"
#include <fstream>
#include <memory>

namespace ncv
{
        // utilities to process .mat files
        namespace mat
        {
                /////////////////////////////////////////////////////////////////////////////////////////

                enum class data_type : int
                {
                        miINT8 = 1,
                        miUINT8 = 2,
                        miINT16 = 3,
                        miUINT16 = 4,
                        miINT32 = 5,
                        miUINT32 = 6,
                        miSINGLE = 7,
                        miDOUBLE = 9,
                        miINT64 = 12,
                        miUINT64 = 13,
                        miMATRIX = 14,
                        miCOMPRESSED = 15,
                        miUTF8 = 16,
                        miUTF16 = 17,
                        miUTF32 = 18,

                        miUNKNOWN
                };

                /////////////////////////////////////////////////////////////////////////////////////////

                string_t to_string(const data_type& type)
                {
                        if (type == data_type::miINT8) return "miINT8";
                        else if (type == data_type::miUINT8) return "miUINT8";
                        else if (type == data_type::miINT16) return "miINT16";
                        else if (type == data_type::miUINT16) return "miUINT16";
                        else if (type == data_type::miINT32) return "miINT32";
                        else if (type == data_type::miUINT32) return "miUINT32";
                        else if (type == data_type::miSINGLE) return "miSINGLE";
                        else if (type == data_type::miDOUBLE) return "miDOUBLE";
                        else if (type == data_type::miINT64) return "miINT64";
                        else if (type == data_type::miUINT64) return "miUINT64";
                        else if (type == data_type::miMATRIX) return "miMATRIX";
                        else if (type == data_type::miCOMPRESSED) return "miCOMPRESSED";
                        else if (type == data_type::miUTF8) return "miUTF8";
                        else if (type == data_type::miUTF16) return "miUTF16";
                        else if (type == data_type::miUTF32) return "miUTF32";
                        else return "miUNKNOWN";
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                size_t to_bytes(const data_type& type)
                {
                        if (type == data_type::miINT8) return 1;
                        else if (type == data_type::miUINT8) return 1;
                        else if (type == data_type::miINT16) return 2;
                        else if (type == data_type::miUINT16) return 2;
                        else if (type == data_type::miINT32) return 4;
                        else if (type == data_type::miUINT32) return 4;
                        else if (type == data_type::miSINGLE) return 4;
                        else if (type == data_type::miDOUBLE) return 8;
                        else if (type == data_type::miINT64) return 8;
                        else if (type == data_type::miUINT64) return 8;
                        else if (type == data_type::miMATRIX) return 0;
                        else if (type == data_type::miCOMPRESSED) return 0;
                        else if (type == data_type::miUTF8) return 0;
                        else if (type == data_type::miUTF16) return 0;
                        else if (type == data_type::miUTF32) return 0;
                        else return 0;
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tint
                >
                data_type make_data_type(tint code)
                {
                        if (code == 1) return data_type::miINT8;
                        else if (code == 2) return data_type::miUINT8;
                        else if (code == 3) return data_type::miINT16;
                        else if (code == 4) return data_type::miUINT16;
                        else if (code == 5) return data_type::miINT32;
                        else if (code == 6) return data_type::miUINT32;
                        else if (code == 7) return data_type::miSINGLE;
                        else if (code == 9) return data_type::miDOUBLE;
                        else if (code == 12) return data_type::miINT64;
                        else if (code == 13) return data_type::miUINT64;
                        else if (code == 14) return data_type::miMATRIX;
                        else if (code == 15) return data_type::miCOMPRESSED;
                        else if (code == 16) return data_type::miUTF8;
                        else if (code == 17) return data_type::miUTF16;
                        else if (code == 18) return data_type::miUTF32;
                        else return data_type::miUNKNOWN;
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                static u_int32_t make_uint32(const u_int8_t* data)
                {
                        return *reinterpret_cast<const u_int32_t*>(data);
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                struct section_t
                {
                        section_t(size_t begin = 0)
                                :       m_begin(begin), m_end(begin),
                                        m_dbegin(begin), m_dend(begin),
                                        m_dtype(data_type::miUNKNOWN)
                        {
                        }

                        bool load(size_t offset, size_t end, u_int32_t dtype, u_int32_t bytes)
                        {
                                // small data format
                                if ((dtype >> 16) != 0)
                                {
                                        m_begin = offset;
                                        m_end = offset + 8;

                                        m_dbegin = offset + 4;
                                        m_dend = offset + 8;

                                        m_dtype = make_data_type((dtype << 16) >> 16);
                                }

                                // regular format
                                else
                                {
                                        m_begin = offset;
                                        m_end = offset + ((make_data_type(dtype) == data_type::miCOMPRESSED) ?
                                                (8 + bytes) :
                                                (8 + bytes + ((8 - bytes) % 8)));

                                        m_dbegin = offset + 8;
                                        m_dend = offset + 8 + bytes;

                                        m_dtype = make_data_type(dtype);
                                }

                                return m_end <= end;
                        }

                        bool load(std::ifstream& istream)
                        {
                                u_int32_t dtype, bytes;
                                return  istream.read(reinterpret_cast<char*>(&dtype), sizeof(u_int32_t)) &&
                                        istream.read(reinterpret_cast<char*>(&bytes), sizeof(u_int32_t)) &&
                                        load(0, std::numeric_limits<size_t>::max(), dtype, bytes);
                        }

                        bool load(const std::vector<u_int8_t>& data, size_t offset = 0)
                        {
                                return  offset + 8 <= data.size() &&
                                        load(offset, data.size(),
                                             make_uint32(&data[offset + 0]), make_uint32(&data[offset + 4]));
                        }

                        bool load(const std::vector<u_int8_t>& data, const section_t& prv)
                        {
                                return load(data, prv.m_end);
                        }

                        size_t begin() const { return m_begin; }
                        size_t end() const { return m_end; }
                        size_t size() const { return end() - begin(); }

                        size_t dbegin() const { return m_dbegin; }
                        size_t dend() const { return m_dend; }
                        size_t dsize() const { return dend() - dbegin(); }

                        size_t          m_begin, m_end;         ///< byte range of the whole section
                        size_t          m_dbegin, m_dend;       ///< byte range of the data section
                        data_type       m_dtype;
                };

                typedef std::vector<section_t>  sections_t;

                /////////////////////////////////////////////////////////////////////////////////////////

                struct array_t
                {
                        array_t()
                        {
                        }

                        bool load(const std::vector<u_int8_t>& data)
                        {
                                // read & check header
                                section_t header;
                                if (!header.load(data))
                                {
                                        log_error() << "failed to load array!";
                                        return false;
                                }

                                if (header.m_dtype != data_type::miMATRIX)
                                {
                                        log_error() << "invalid array type: expecting "
                                                    << mat::to_string(data_type::miMATRIX) << "!";
                                        return false;
                                }

                                if (header.end() != data.size())
                                {
                                        log_error() << "invalid array size in bytes!";
                                        return false;
                                }

                                log_info() << "array header: dtype = " << mat::to_string(header.m_dtype)
                                           << ", bytes = " << header.size() << "/" << data.size() << ".";

                                // read & check sections
                                m_sections.clear();

                                for (size_t i = 8; i < data.size(); )
                                {
                                        section_t section;
                                        if (!section.load(data, i))
                                        {
                                                break;
                                        }

                                        log_info() << "array section: dtype = " << mat::to_string(section.m_dtype)
                                                   << ", range = [" << section.begin() << ", " << section.end()
                                                   << "], drange = [" << section.dbegin() << ", " << section.dend()
                                                   << "], bytes = " << section.dsize() << "/" << section.size() << ".";

                                        m_sections.push_back(section);
                                        i = section.end();
                                }

                                if (m_sections.size() != 4)
                                {
                                        log_error() << "invalid array sections! expecting 4 sections!";
                                        return false;
                                }

                                // decode sections:
                                //      first:  flags + class
                                //      second: dimensions
                                //      third:  name
//                                const mat::section_t& sect1 = m_sections[0];
                                const mat::section_t& sect2 = m_sections[1];
                                const mat::section_t& sect3 = m_sections[2];
                                const mat::section_t& sect4 = m_sections[3];

                                m_name = string_t(data.begin() + sect3.dbegin(),
                                                  data.begin() + sect3.dend());

                                m_dims.clear();
                                size_t values = 1;

                                for (size_t i = sect2.dbegin(); i < sect2.dend(); i += 4)
                                {
                                        const size_t dim = make_uint32(&data[i]);
                                        m_dims.push_back(dim);
                                        values *= dim;
                                }

                                // check bytes
                                if (values * to_bytes(sect4.m_dtype) != sect4.dsize())
                                {
                                        log_error() << "invalid array sections! mismatching number of bytes!";
                                        return false;
                                }

                                // OK
                                return true;
                        }

                        void log(logger_t& logger) const
                        {
                                logger << "sections = " << m_sections.size()
                                       << ", name = " << m_name
                                       << ", dims = ";
                                for (size_t i = 0; i < m_dims.size(); i ++)
                                {
                                        logger << m_dims[i] << ((i + 1 == m_dims.size()) ? "" : "x");
                                }
                        }

                        indices_t       m_dims;
                        string_t        m_name;
                        sections_t      m_sections;
                };
        }

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
                        mat::section_t section;
                        if (!section.load(istream))
                        {
                                log_error() << "SVHN: failed to read section!";
                                return 0;
                        }

                        if (section.m_dtype != mat::data_type::miCOMPRESSED)
                        {
                                log_error() << "SVHN: invalid data type <" << mat::to_string(section.m_dtype)
                                            << ">! expecting " << mat::to_string(mat::data_type::miCOMPRESSED) << "!";
                                return 0;
                        }

                        log_info() << "SVHN: uncompressing " << section.dsize() << " bytes ...";

                        std::vector<u_int8_t>& data = (isection == 0) ? image_data : label_data;
                        if (!ncv::zuncompress(istream, section.dsize(), data))
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

        bool svhn_task_t::build_folds(size_t n_train, size_t n_test)
        {
                const fold_t train_fold = std::make_pair(0, protocol::train);
                m_folds[train_fold] = make_samples(0, n_train, sample_region(0, 0));

                const fold_t test_fold = std::make_pair(0, protocol::test);
                m_folds[test_fold] = make_samples(n_train, n_test, sample_region(0, 0));

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t svhn_task_t::decode(
                const std::vector<u_int8_t>& idata,
                const std::vector<u_int8_t>& ldata,
                protocol p)
        {
                // decode image & label arrays
                mat::array_t iarray, larray;
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

                const mat::section_t& isection = iarray.m_sections[3];
                const mat::section_t& lsection = larray.m_sections[3];

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
                if (    isection.m_dtype != mat::data_type::miUINT8 ||
                        lsection.m_dtype != mat::data_type::miUINT8)
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
