#include "image.h"
#include "text.h"
#include "util/math.hpp"
#include <fstream>
#include <boost/algorithm/string.hpp>

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/gil/extension/io/tiff_io.hpp>
#include <boost/gil/extension/io/png_io.hpp>

namespace ncv
{
        enum class imagetype : int
        {
                jpeg,
                png,
                tif,
                pgm,
                unknown
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        imagetype decode_image_type(const string_t& path)
        {
                if (text::iends_with(path, ".jpg") || text::iends_with(path, ".jpeg"))
                {
                        return imagetype::jpeg;
                }

                else if (text::iends_with(path, ".png"))
                {
                        return imagetype::png;
                }

                else if (text::iends_with(path, ".tif") || text::iends_with(path, ".tiff"))
                {
                        return imagetype::tif;
                }

                else if (text::iends_with(path, ".pgm"))
                {
                        return imagetype::pgm;
                }

                else
                {
                        return imagetype::unknown;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool load_rgba(const string_t& path, rgba_matrix_t& rgba)
        {
                const imagetype itype = decode_image_type(path);
                switch (itype)
                {
                case imagetype::png:    // boost::gil decoding
                case imagetype::jpeg:
                case imagetype::tif:
                        {
                                boost::gil::argb8_image_t image;

                                if (itype == imagetype::png)
                                {
                                        boost::gil::png_read_and_convert_image(path, image);
                                }
                                else if (itype == imagetype::jpeg)
                                {
                                        boost::gil::jpeg_read_and_convert_image(path, image);
                                }
                                else
                                {
                                        boost::gil::tiff_read_and_convert_image(path, image);
                                }

                                const int rows = static_cast<int>(image.height());
                                const int cols = static_cast<int>(image.width());
                                rgba.resize(rows, cols);

                                const boost::gil::argb8_image_t::const_view_t view = boost::gil::const_view(image);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                const boost::gil::argb8_pixel_t pixel = view(c, r);
                                                rgba(r, c) = color::make_rgba(pixel[1], pixel[2], pixel[3], pixel[0]);
                                        }
                                }
                        }

                case imagetype::pgm:    // PGM binary decoding
                        {
                                std::ifstream is(path);

                                // read header
                                string_t line_type, line_size, line_maxv;
                                if (    !is.is_open() ||
                                        !std::getline(is, line_type) ||
                                        !std::getline(is, line_size) ||
                                        !std::getline(is, line_maxv) ||
                                        line_type != "P5" ||
                                        line_maxv != "255")
                                {
                                        return false;
                                }

                                strings_t tokens;
                                text::split(tokens, line_size, text::is_any_of(" "));

                                int rows = -1, cols = -1;
                                if (    tokens.size() != 2 ||
                                        (cols = text::from_string<int>(tokens[0])) < 1 ||
                                        (rows = text::from_string<int>(tokens[1])) < 1)
                                {
                                        return false;
                                }

                                // read pixels
                                std::vector<u_int8_t> grays(rows * cols);
                                if (!is.read((char*)(grays.data()), grays.size()))
                                {
                                        return false;
                                }

                                rgba.resize(rows, cols);
                                for (int r = 0, i = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++, i ++)
                                        {
                                                const rgba_t gray = grays[i];
                                                rgba(r, c) = color::make_rgba(gray, gray, gray);
                                        }
                                }
                        }
                        return true;

                case imagetype::unknown:
                default:
                        return false;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool save_rgba(const string_t& path, const rgba_matrix_t& rgba)
        {
                const int rows = math::cast<int>(rgba.rows());
                const int cols = math::cast<int>(rgba.cols());

                const imagetype itype = decode_image_type(path);
                switch (itype)
                {
                case imagetype::png: // boost::gil RGBA encoding
                        {
                                boost::gil::argb8_image_t image(cols, rows);

                                boost::gil::argb8_image_t::view_t view = boost::gil::view(image);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                boost::gil::argb8_pixel_t& pixel = view(c, r);
                                                pixel[0] = color::make_alpha(rgba(r, c));
                                                pixel[1] = color::make_red(rgba(r, c));
                                                pixel[2] = color::make_green(rgba(r, c));
                                                pixel[3] = color::make_blue(rgba(r, c));
                                        }
                                }

                                boost::gil::png_write_view(path, view);
                                return true;
                        }

                case imagetype::jpeg: // boost::gil RGB encoding
                case imagetype::tif:
                        {
                                boost::gil::rgb8_image_t image(cols, rows);

                                boost::gil::rgb8_image_t::view_t view = boost::gil::view(image);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                boost::gil::rgb8_pixel_t& pixel = view(c, r);
                                                pixel[0] = color::make_red(rgba(r, c));
                                                pixel[1] = color::make_green(rgba(r, c));
                                                pixel[2] = color::make_blue(rgba(r, c));
                                        }
                                }

                                if (itype == imagetype::jpeg)
                                {
                                        boost::gil::jpeg_write_view(path, view);
                                }
                                else
                                {
                                        boost::gil::tiff_write_view(path, view);
                                }
                                return true;
                        }

                case imagetype::pgm:    // PGM binary encoding
                        {
                                std::ofstream os(path);

                                // write header
                                if (    !os.is_open() ||
                                        !(os << "P5" << std::endl) ||
                                        !(os << cols << " " << rows << std::endl) ||
                                        !(os << "255" << std::endl))
                                {
                                        return false;
                                }

                                // write pixels
                                std::vector<u_int8_t> grays(rows * cols);
                                for (int r = 0, i = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++, i ++)
                                        {
                                                grays[i] = static_cast<u_int8_t>(color::make_luma(rgba(r, c)));
                                        }
                                }

                                return os.write((const char*)(grays.data()), grays.size());
                        }

                case imagetype::unknown:
                default:
                        return false;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool image_t::load(const string_t& path)
        {
                return ncv::load_rgba(path, m_rgba);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool image_t::load_gray(const char* buffer, size_t rows, size_t cols)
        {
                m_rgba.resize(rows, cols);

                for (size_t y = 0, i = 0; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, i ++)
                        {
                                m_rgba(y, x) = color::make_rgba(buffer[i], buffer[i], buffer[i]);
                        }
                }

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool image_t::load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride)
        {
                m_rgba.resize(rows, cols);

                for (size_t y = 0, dr = 0, dg = dr + stride, db = dg + stride; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, dr ++, dg ++, db ++)
                        {
                                m_rgba(y, x) = color::make_rgba(buffer[dr], buffer[dg], buffer[db]);
                        }
                }

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t image_t::make_target(const rect_t& region) const
        {
                // load the target of the most overlapping annotation
                scalar_t best_overlap = 0.0;
                size_t best_index = 0;

                for (size_t i = 0; i < m_annotations.size(); i ++)
                {
                        const annotation_t& anno = m_annotations[i];

                        const scalar_t overlap = geom::overlap(anno.m_region, region);
                        if (overlap > best_overlap)
                        {
                                best_overlap = overlap;
                                best_index = i;
                        }
                }

                static const scalar_t thres_overlap = 0.80;
                if (    best_overlap > thres_overlap &&
                        best_index < m_annotations.size())
                {
                        return m_annotations[best_index].m_target;
                }
                else
                {
                        return vector_t();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
