#include "task_svhn.h"
#include "common/math.hpp"
#include "common/logger.h"
#include "loss.h"
#include <fstream>
#include <memory>
#include <zlib.h>

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
                        return  ((u_int32_t)(data[0]) << 0) |
                                ((u_int32_t)(data[1]) << 8) |
                                ((u_int32_t)(data[2]) << 16) |
                                ((u_int32_t)(data[3]) << 24);
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                struct section_t
                {
                        section_t(size_t offset = 0)
                                :       m_offset(offset), m_bytes(0),
                                        m_type(0),
                                        m_dtype(data_type::miUNKNOWN)
                        {
                        }

                        bool load(size_t offset, u_int32_t type, u_int32_t bytes)
                        {
                                m_offset = offset;
                                m_bytes = bytes;
                                m_type = type;
                                m_dtype = make_data_type(type);
                                return true;
                        }

                        bool load(std::ifstream& istream)
                        {
                                u_int32_t data_type, num_bytes;
                                if (    !istream.read(reinterpret_cast<char*>(&data_type), sizeof(u_int32_t)) ||
                                        !istream.read(reinterpret_cast<char*>(&num_bytes), sizeof(u_int32_t)))
                                {
                                        return false;
                                }
                                else
                                {
                                        return load(0, data_type, num_bytes);
                                }
                        }

                        bool load(const std::vector<u_int8_t>& data, size_t offset = 0)
                        {
                                return  offset + 8 <= data.size() &&
                                        load(offset, make_uint32(&data[offset + 0]), make_uint32(&data[offset + 4]));
                        }

                        bool load(const std::vector<u_int8_t>& data, const section_t& prv)
                        {
                                return load(data, prv.m_offset + prv.m_bytes + 8);
                        }

                        size_t begin() const { return m_offset + 8; }
                        size_t end() const { return begin() + size(); }
                        size_t size() const { return m_bytes; }

                        size_t          m_offset;
                        size_t          m_bytes;
                        u_int32_t       m_type;
                        data_type       m_dtype;
                };

                typedef std::vector<section_t>  sections_t;

                /////////////////////////////////////////////////////////////////////////////////////////

                struct array_t
                {
                        array_t() : m_flags(0), m_class(0)
                        {
                        }

                        bool load(const std::vector<u_int8_t>& data)
                        {
                                // read & check header
                                section_t header;
                                if (!header.load(data))
                                {
                                        log_info() << "failed to load array!";
                                        return false;
                                }

                                if (header.m_dtype != data_type::miMATRIX)
                                {
                                        log_info() << "invalid array type <" << header.m_type
                                                   << ">! expecting " << mat::to_string(data_type::miMATRIX) << "!";
                                        return false;
                                }

                                if (header.m_bytes + 8 != data.size())
                                {
                                        log_info() << "invalid array size in bytes!";
                                        return false;
                                }

                                log_info() << "array header: type = " << header.m_type
                                           << "/" << mat::to_string(header.m_dtype)
                                           << ", bytes = " << header.m_bytes;

                                // read & check sections
                                m_sections.clear();

                                mat::section_t prv_section, crt_section;
                                while (crt_section.load(data, prv_section))
                                {
                                        log_info() << "array section: type = " << crt_section.m_type
                                                  << "/" << mat::to_string(crt_section.m_dtype)
                                                  << ", size = " << crt_section.size()
                                                  << ", range = [" << crt_section.begin()
                                                  << ", " << crt_section.end() << "]";
                                        m_sections.push_back(crt_section);
                                        prv_section = crt_section;
                                }

                                if (m_sections.size() < 4)
                                {
                                        log_info() << "invalid array sections! expecting at least 4 sections!";
                                        return false;
                                }

                                if (m_sections[0].m_bytes != 8)
                                {
                                        log_info() << "invalid array's first section! expecting 8 bytes structure!";
                                        return false;
                                }

                                // decode sections:
                                //      first:  flags + class
                                //      second: dimensions
                                //      third:  name
                                const mat::section_t& sect1 = m_sections[0];
                                const mat::section_t& sect2 = m_sections[1];
//                                const mat::section_t& sect3 = m_sections[2];

                                // TODO: read correctly the name!

                                m_flags = data[sect1.begin() + 2];
                                m_class = data[sect1.begin() + 3];

                                m_dims.clear();
                                for (size_t i = 0; i < sect2.size(); i += 4)
                                {
                                        m_dims.push_back(make_uint32(&data[sect2.begin() + i]));
                                }

                                return true;
                        }

                        void log(logger_t& logger) const
                        {
                                logger << "sections = " << m_sections.size()
                                       << ", class = " << m_class
                                       << ", flags = " << m_flags
                                       << ", size = ";
                                for (size_t i = 0; i < m_dims.size(); i ++)
                                {
                                        logger << m_dims[i] << ((i + 1 == m_dims.size()) ? "" : "x");
                                }
                        }

                        u_int32_t       m_flags, m_class;
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

                        log_info() << "SVHN: uncompressing " << section.m_bytes << " bytes ...";

                        // zlib decompression buffers
                        static const u_int32_t CHUNK = 64 * 1024;

                        z_stream strm;
                        unsigned char in[CHUNK];
                        unsigned char out[CHUNK];

                        strm.zalloc = Z_NULL;
                        strm.zfree = Z_NULL;
                        strm.opaque = Z_NULL;
                        strm.avail_in = 0;
                        strm.next_in = Z_NULL;
                        if (inflateInit(&strm) != Z_OK)
                        {
                                log_error() << "SVHN: cannot initialize zlib!";
                                return 0;
                        }

                        // decompress the data chunk
                        u_int32_t num_bytes = math::cast<u_int32_t>(section.m_bytes);
                        std::vector<u_int8_t>& data = (isection == 0) ? image_data : label_data;
                        for ( ; num_bytes > 0; )
                        {
                                const u_int32_t to_read = num_bytes >= CHUNK ? CHUNK : num_bytes;
                                num_bytes -= to_read;

                                if (!istream.read(reinterpret_cast<char*>(in), to_read))
                                {
                                        inflateEnd(&strm);
                                        log_error() << "SVHN: failed to read compressed data!";
                                        return 0;
                                }

                                strm.avail_in = to_read;
                                strm.next_in = in;

                                do
                                {
                                        strm.avail_out = CHUNK;
                                        strm.next_out = out;

                                        const int ret = inflate(&strm, Z_NO_FLUSH);
                                        if (ret != Z_OK && ret != Z_STREAM_END)
                                        {
                                                inflateEnd(&strm);
                                                log_error() << "SVHN: decompression failed!";
                                                return 0;
                                        }

                                        const u_int32_t have = CHUNK - strm.avail_out;
                                        data.insert(data.end(), out, out + have);
                                }
                                while (strm.avail_out == 0);
                        }

                        inflateEnd(&strm);

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
                const std::vector<u_int8_t>& image_data,
                const std::vector<u_int8_t>& label_data,
                protocol p)
        {
                // decode image array
                mat::array_t image_array;
                if (!image_array.load(image_data))
                {
                        log_error() << "SVHN: invalid image array!";
                        return false;
                }

                image_array.log(log_info() << "SVHN: image array: ");

                // decode label array
                mat::array_t label_array;
                if (!label_array.load(label_data))
                {
                        log_error() << "SVHN: invalid label array!";
                        return false;
                }

                label_array.log(log_info() << "SVHN: label array: ");

                // check array size
                if (    image_array.m_dims.size() != 4 ||
                        image_array.m_dims[0] != n_rows() ||
                        image_array.m_dims[1] != n_cols() ||
                        image_array.m_dims[2] != 3 ||

                        label_array.m_dims.size() != 2 ||
                        label_array.m_dims[1] != 1 ||

                        image_array.m_dims[3] != label_array.m_dims[0])
                {
                        log_error() << "SVHN: invalid or mis-matching image & label array size!";
                        return false;
                }

                // load images & labels
                const size_t n_samples = image_array.m_dims[3];
                const size_t image_size = n_rows() * n_cols() * 3;

                size_t loaded_images = 0;
                for (mat::sections_t::const_iterator it = image_array.m_sections.begin() + 3;
                        it != image_array.m_sections.end(); ++ it)
                {
                        const mat::section_t& section = *it;

                        std::cout << "section = [" << section.begin() << ", " << section.end()
                                  << "], data = " << image_data.size() << std::endl;

                        for (size_t i = section.begin(); i + image_size < section.end(); i += image_size)
                        {
                                if (i + image_size >= image_data.size())
                                {
                                        std::cout << "!!!! delta = " << (i + image_size - image_data.size()) << std::endl;
                                        break;
                                }

//                                image_t image;
//                                image.m_protocol = p;
//                                image.load_rgba((char*)&image_data[i], n_rows(), n_cols(), 1);

//                                TODO: it crashes here!

//                                m_images.push_back(image);
                                loaded_images ++;
                        }
                }

                std::cout << "loaded " << loaded_images << "/" << n_samples << std::endl;

                for (size_t i = 0; i < n_samples; i ++)
                {

                }

                // load images & labels

                size_t cnt = 0;


//                char label[1];

//                // load images and annotations
//                size_t cnt = 0;
//                while ( istream.read(label, 1) &&
//                        istream.read(buffer, sizeof(buffer)))
//                {
//                        const size_t ilabel = math::cast<size_t>(label[0]);
//                        if (ilabel >= n_outputs())
//                        {
//                                continue;
//                        }

//                        const annotation_t anno(sample_region(0, 0),
//                                "digit" + text::to_string(ilabel),
//                                ncv::class_target(ilabel, n_outputs()));

//                        image_t image;
//                        image.m_protocol = p;
//                        image.m_annotations.push_back(anno);
//                        image.load_rgba(buffer, n_rows(), n_cols());

//                        m_images.push_back(image);
//                        ++ cnt;
//                }

                return cnt;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
