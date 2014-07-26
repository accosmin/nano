#include "image.h"
#include "common/math.hpp"
#include "common/bilinear.hpp"
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

        static bool load_image(const string_t& path,
                color_mode mode, rgba_matrix_t& rgba, luma_matrix_t& luma)
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

                                const boost::gil::argb8_image_t::const_view_t view = boost::gil::const_view(image);
                                switch (mode)
                                {
                                case color_mode::luma:
                                        luma.resize(rows, cols);
                                        for (int r = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++)
                                                {
                                                        const boost::gil::argb8_pixel_t pix = view(c, r);
                                                        luma(r, c) = color::make_luma(pix[1], pix[2], pix[3]);
                                                }
                                        }
                                        break;

                                case color_mode::rgba:
                                        rgba.resize(rows, cols);
                                        for (int r = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++)
                                                {
                                                        const boost::gil::argb8_pixel_t pix = view(c, r);
                                                        rgba(r, c) = color::make_rgba(pix[1], pix[2], pix[3], pix[0]);
                                                }
                                        }
                                        break;
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

                                switch (mode)
                                {
                                case color_mode::luma:
                                        luma.resize(rows, cols);
                                        for (int r = 0, i = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++, i ++)
                                                {
                                                        luma(r, c) = grays[i];
                                                }
                                        }
                                        break;

                                case color_mode::rgba:
                                        rgba.resize(rows, cols);
                                        for (int r = 0, i = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++, i ++)
                                                {
                                                        const rgba_t gray = grays[i];
                                                        rgba(r, c) = color::make_rgba(gray, gray, gray);
                                                }
                                        }
                                        break;
                                }
                        }
                        return true;

                case imagetype::unknown:
                default:
                        return false;
                }
        }

        static bool save_image(const string_t& path,
                color_mode mode, const rgba_matrix_t& rgba, const luma_matrix_t& luma)
        {
                const int rows = math::cast<int>(mode == color_mode::rgba ? rgba.rows() : luma.rows());
                const int cols = math::cast<int>(mode == color_mode::rgba ? rgba.cols() : luma.cols());

                const imagetype itype = decode_image_type(path);
                switch (itype)
                {
                case imagetype::png: // boost::gil RGBA encoding
                        {
                                boost::gil::argb8_image_t image(cols, rows);

                                boost::gil::argb8_image_t::view_t view = boost::gil::view(image);
                                switch (mode)
                                {
                                case color_mode::luma:
                                        for (int r = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++)
                                                {
                                                        boost::gil::argb8_pixel_t& pixel = view(c, r);
                                                        pixel[0] = 255;
                                                        pixel[1] = luma(r, c);
                                                        pixel[2] = luma(r, c);
                                                        pixel[3] = luma(r, c);
                                                }
                                        }
                                        break;

                                case color_mode::rgba:
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
                                }

                                boost::gil::png_write_view(path, view);
                                return true;
                        }

                case imagetype::jpeg: // boost::gil RGB encoding
                case imagetype::tif:
                        {
                                boost::gil::rgb8_image_t image(cols, rows);

                                boost::gil::rgb8_image_t::view_t view = boost::gil::view(image);
                                switch (mode)
                                {
                                case color_mode::luma:
                                        for (int r = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++)
                                                {
                                                        boost::gil::rgb8_pixel_t& pixel = view(c, r);
                                                        pixel[0] = luma(r, c);
                                                        pixel[1] = luma(r, c);
                                                        pixel[2] = luma(r, c);
                                                }
                                        }
                                        break;

                                case color_mode::rgba:
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
                                        break;
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
                                tensor::matrix_types_t<u_int8_t>::tmatrix grays(rows * cols);
                                switch (mode)
                                {
                                case color_mode::luma:
                                        grays
                                        for (int r = 0, i = 0; r < rows; r ++)
                                        {
                                                for (int c = 0; c < cols; c ++, i ++)
                                                {
                                                        grays[i] = static_cast<u_int8_t>(color::make_luma(rgba(r, c)));
                                                }
                                        }
                                        break;

                                        case color_
                                }

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
        inline matrix_t load_luma(const luma_matrix_t& luma, const rect_t& region)
        {
                return detail::make_data(luma, region, [] (luma_t g) { return g; });
        }

        image_t::image_t(size_t rows, size_t cols, color_mode mode)
                :       m_rows(rows),
                        m_cols(cols),
                        m_mode(mode)
        {
                resize(rows, cols, mode);
        }

        void image_t::resize(size_t rows, size_t cols, color_mode mode)
        {
                m_mode = mode;
                m_rows = rows;
                m_cols = cols;

                if (is_luma())
                {
                        m_luma.resize(rows, cols);
                        m_rgba.resize(0, 0);
                }

                else if (is_rgba())
                {
                        m_luma.resize(0, 0);
                        m_rgba.resize(rows, cols);
                }
        }

        bool image_t::load_rgba(const string_t& path)
        {

        }

        bool image_t::load_luma(const string_t& path)
        {

        }

        bool image_t::load_luma(const char* buffer, size_t rows, size_t cols)
        {

        }

        bool image_t::load_rgba(const char* buffer, size_t rows, size_t cols)
        {

        }

        bool image_t::load_rgba(const char* buffer, size_t rows, size_t cols, size_t stride)
        {

        }

        bool image_t::load_rgba(const rgba_matrix_t& data)
        {

        }

        bool image_t::load_luma(const rgba_matrix_t& data)
        {

        }

        bool image_t::load_luma(const luma_matrix_t& data)
        {

        }

        bool image_t::from_tensor(const tensor_t& data) const
        {

        }

        bool image_t::from_tensor(const tensor_t& data, const rect_t& region) const
        {

        }

        bool image_t::save(const string_t& path) const
        {

        }

        tensor_t image_t::to_tensor() const
        {

        }

        tensor_t image_t::to_tensor(const rect_t& region) const
        {

        }

        void image_t::make_rgba()
        {
                if (is_luma())
                {
                        m_rgba.resize(rows(), cols());
                        math::transform(m_luma, m_rgba, color::make_rgba);
                        m_luma.resize(0, 0);
                }

                else if (is_rgba())
                {
                }
        }

        void image_t::make_luma()
        {
                if (is_luma())
                {
                }

                else if (is_rgba())
                {
                        m_luma.resize(rows(), cols());
                        math::transform(m_rgba, m_luma, color::make_luma);
                        m_rgba.resize(0, 0);
                }
        }

        void image_t::fill(rgba_t rgba) const
        {

        }

        void image_t::fill(luma_t luma) const
        {

        }

        bool image_t::copy(coord_t r, coord_t c, const rgba_matrix_t& patch) const
        {

        }

        bool image_t::copy(coord_t r, coord_t c, const luma_matrix_t& patch) const
        {

        }

        bool image_t::copy(coord_t r, coord_t c, const image_t& patch) const
        {

        }

        bool image_t::copy(coord_t r, coord_t c, const image_t& patch, const rect_t& region) const
        {

        }

        bool image_t::set(coord_t r, coord_t c, rgba_t rgba)
        {

        }

        bool image_t::set(coord_t r, coord_t c, luma_t gray)
        {
                if (    r >= 0 && r < static_cast<coord_t>(rows()) &&
                        c >= 0 && c < static_cast<coord_t>(cols()))
                {

                }

                else
                {
                        return false;
                }
        }

        void image_t::transpose_in_place()
        {
                m_rgba.transposeInPlace();
                m_luma.transposeInPlace();
        }

        void image_t::scale(scalar_t factor)
        {
                if (is_luma())
                {
                        luma_matrix_t luma_scaled;
                        math::bilinear(m_luma, luma_scaled, factor);

                        m_luma = luma_scaled;
                        m_rows = static_cast<size_t>(m_luma.rows());
                        m_cols = static_cast<size_t>(m_luma.cols());
                }

                else if (is_rgba())
                {
                        cielab_matrix_t cielab, cielab_scaled;
                        cielab.resize(rows(), cols());
                        math::transform(m_rgba, cielab, color::make_cielab);

                        math::bilinear(cielab, cielab_scaled, factor);

                        m_rgba.resize(cielab_scaled.rows(), cielab_scaled.cols());
                        math::transform(cielab_scaled, m_rgba, [] (const cielab_t& lab) { return color::make_rgba(lab); });
                        m_rows = static_cast<size_t>(m_rgba.rows());
                        m_cols = static_cast<size_t>(m_rgba.cols());
                }
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
