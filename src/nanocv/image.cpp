#include "image.h"
#include "common/math.hpp"
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

        static imagetype decode_image_type(const string_t& path)
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

        bool load_rgba(const tensor_t& data, rgba_matrix_t& rgba)
        {
                if (data.dims() == 1)
                {
                        const auto gmap = data.plane_matrix(0);

                        rgba.resize(data.rows(), data.cols());
                        for (size_t r = 0; r < data.rows(); r ++)
                        {
                                for (size_t c = 0; c < data.cols(); c ++)
                                {
                                        const rgba_t gray = math::cast<rgba_t>(gmap(r, c) * 255.0) & 0xFF;
                                        rgba(r, c) = color::make_rgba(gray, gray, gray);
                                }
                        }

                        return true;
                }

                else if (data.dims() == 3)
                {
                        const auto rmap = data.plane_matrix(0);
                        const auto gmap = data.plane_matrix(1);
                        const auto bmap = data.plane_matrix(2);

                        rgba.resize(data.rows(), data.cols());
                        for (size_t r = 0; r < data.rows(); r ++)
                        {
                                for (size_t c = 0; c < data.cols(); c ++)
                                {
                                        const rgba_t red = math::cast<rgba_t>(rmap(r, c) * 255.0) & 0xFF;
                                        const rgba_t green = math::cast<rgba_t>(gmap(r, c) * 255.0) & 0xFF;
                                        const rgba_t blue = math::cast<rgba_t>(bmap(r, c) * 255.0) & 0xFF;
                                        rgba(r, c) = color::make_rgba(red, green, blue);
                                }
                        }

                        return true;
                }

                else
                {
                        return false;
                }
        }

        bool load_gray(const char* buffer, size_t rows, size_t cols, rgba_matrix_t& rgba)
        {
                rgba.resize(rows, cols);

                for (size_t y = 0, i = 0; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, i ++)
                        {
                                rgba(y, x) = color::make_rgba(buffer[i], buffer[i], buffer[i]);
                        }
                }

                return true;
        }

        bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride, rgba_matrix_t& rgba)
        {
                rgba.resize(rows, cols);

                for (size_t y = 0, dr = 0, dg = dr + stride, db = dg + stride; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, dr ++, dg ++, db ++)
                        {
                                rgba(y, x) = color::make_rgba(buffer[dr], buffer[dg], buffer[db]);
                        }
                }

                return true;
        }

        ///
        /// \brief save RGBA image to disk
        ///
        bool save_rgba(const string_t& path, const rgba_matrix_t& rgba);

        ///
        /// \brief load RGBA image from disk
        ///
        bool load_rgba(const string_t& path, rgba_matrix_t& rgba);

        ///
        /// \brief load gray/color image from [0, 1] normalized tensor
        ///
        bool load_rgba(const tensor_t& tensor, rgba_matrix_t& rgba);

        ///
        /// \brief load gray image from buffer
        ///
        bool load_gray(const char* buffer, size_t rows, size_t cols, rgba_matrix_t&);
        bool load_gray(const char* buffer, size_t rows, size_t cols, gray_matrix_t&);

        ///
        /// \brief load RGBA image from buffer
        ///
        bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride, rgba_matrix_t&);
        bool load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride, gray_matrix_t&);

        namespace detail
        {
                ///
                /// \brief retrieve the [0, 1] normalized color channel
                ///
                template
                <
                        typename timage,
                        typename toperator
                >
                matrix_t make_data(const timage& image, const rect_t& region, const toperator& op)
                {
                        const coord_t top = geom::top(region), left = geom::left(region);
                        const coord_t rows = geom::rows(region), cols = geom::cols(region);
                        const scalar_t scale = 1.0 / 255.0;

                        matrix_t data(rows, cols);

                        if (    top == 0 && left == 0 &&
                                rows == static_cast<coord_t>(image.rows()) &&
                                cols == static_cast<coord_t>(image.cols()))
                        {
                                const coord_t size = rows * cols;

                                for (coord_t i = 0; i < size; i ++)
                                {
                                        data(i) = scale * op(image(i));
                                }
                        }

                        else
                        {
                                for (coord_t r = 0; r < rows; r ++)
                                {
                                        for (coord_t c = 0; c < cols; c ++)
                                        {
                                                data(r, c) = scale * op(image(top + r, left + c));
                                        }
                                }
                        }

                        return data;
                }
        }

        ///
        /// \brief retrieve the scaled [0, 1] RGB input vector
        ///
        inline matrix_t load_red(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_red);
        }
        inline matrix_t load_green(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_green);
        }
        inline matrix_t load_blue(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_blue);
        }
        inline matrix_t load_luma(const rgba_matrix_t& rgba, const rect_t& region)
        {
                return detail::make_data(rgba, region, color::make_luma);
        }
        inline matrix_t load_luma(const gray_matrix_t& gray, const rect_t& region)
        {
                return detail::make_data(gray, region, [] (gray_t g) { return g; });
        }



//        switch (m_color)
//        {
//        case color_mode::luma:
//                data.resize(1, irows(), icols());
//                data.copy_plane_from(0, load_luma(image, region));
//                break;

//        case color_mode::rgba:
//                data.resize(3, irows(), icols());
//                data.copy_plane_from(0, load_red(image, region));
//                data.copy_plane_from(1, load_green(image, region));
//                data.copy_plane_from(2, load_blue(image, region));
//                break;
//        }

//        return data;
}
