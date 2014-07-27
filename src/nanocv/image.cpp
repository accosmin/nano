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
                                luma_matrix_t grays(rows, cols);
                                switch (mode)
                                {
                                case color_mode::luma:
                                        grays = luma;
                                        break;

                                case color_mode::rgba:
                                        math::transform(rgba, grays, [] (rgba_t c) { return color::make_luma(c); });
                                        break;
                                }

                                return os.write(reinterpret_cast<const char*>(grays.data()),
                                                static_cast<std::streamsize>(grays.size()));
                        }

                case imagetype::unknown:
                default:
                        return false;
                }
        }

        image_t::image_t(coord_t rows, coord_t cols, color_mode mode)
                :       m_rows(rows),
                        m_cols(cols),
                        m_mode(mode)
        {
                resize(rows, cols, mode);
        }

        void image_t::resize(coord_t rows, coord_t cols, color_mode mode)
        {
                m_mode = mode;
                m_rows = rows;
                m_cols = cols;

                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.resize(rows, cols);
                        m_rgba.resize(0, 0);
                        break;

                case color_mode::rgba:
                        m_luma.resize(0, 0);
                        m_rgba.resize(rows, cols);
                        break;
                }
        }

        bool image_t::setup_rgba()
        {
                m_luma.resize(0, 0);
                m_mode = color_mode::rgba;
                m_rows = static_cast<coord_t>(m_rgba.rows());
                m_cols = static_cast<coord_t>(m_rgba.rows());
                return true;
        }

        bool image_t::setup_luma()
        {
                m_rgba.resize(0, 0);
                m_mode = color_mode::luma;
                m_rows = static_cast<coord_t>(m_luma.rows());
                m_cols = static_cast<coord_t>(m_luma.rows());
                return true;
        }

        bool image_t::load_rgba(const string_t& path)
        {
                return  load_image(path, color_mode::rgba, m_rgba, m_luma) &&
                        setup_rgba();
        }

        bool image_t::load_luma(const string_t& path)
        {
                return  load_image(path, color_mode::luma, m_rgba, m_luma) &&
                        setup_luma();
        }

        bool image_t::load_luma(const char* buffer, coord_t rows, coord_t cols)
        {
                m_luma.resize(rows, cols);
                for (coord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (coord_t c = 0; c < cols; c ++, i ++)
                        {
                                m_luma(r, c) = static_cast<luma_t>(buffer[i]);
                        }
                }

                return setup_luma();
        }

        bool image_t::load_rgba(const char* buffer, coord_t rows, coord_t cols, coord_t stride)
        {
                m_rgba.resize(rows, cols);
                for (coord_t r = 0, dr = 0, dg = dr + stride, db = dg + stride; r < rows; r ++)
                {
                        for (coord_t c = 0; c < cols; c ++, dr ++, dg ++, db ++)
                        {
                                m_rgba(r, c) = color::make_rgba(buffer[dr], buffer[dg], buffer[db]);
                        }
                }

                return setup_rgba();
        }

        bool image_t::load_rgba(const rgba_matrix_t& data)
        {
                m_rgba = data;

                return setup_rgba();
        }

        bool image_t::load_luma(const rgba_matrix_t& data)
        {
                m_luma.resize(data.rows(), data.cols());
                math::transform(data, m_luma, [] (rgba_t c) { return color::make_luma(c); });

                return setup_luma();
        }

        bool image_t::load_luma(const luma_matrix_t& data)
        {
                m_luma = data;

                return setup_luma();
        }

        bool image_t::from_tensor(const tensor_t& data)
        {
                return from_tensor(data, geom::make_rect(0, 0, data.cols(), data.rows()));
        }

        bool image_t::from_tensor(const tensor_t& data, const rect_t& region)
        {
                const coord_t t = geom::top(region), l = geom::left(region);
                const coord_t rows = geom::rows(region), cols = geom::cols(region);
                const coord_t size = rows * cols;

                const coord_t drows = static_cast<coord_t>(data.rows());
                const coord_t dcols = static_cast<coord_t>(data.cols());
                const scalar_t scale = 255.0;

                if (    t < 0 || t + rows >= drows ||
                        l < 0 || l + cols >= dcols)
                {
                        return false;
                }

                const bool use_all = t == 0 && l == 0 && rows == drows && cols == dcols;

                if (data.dims() == 1)
                {
                        const auto gmap = data.plane_matrix(0);

                        m_luma.resize(rows, cols);
                        if (use_all)
                        {
                                for (coord_t i = 0; i < size; i ++)
                                {
                                        const rgba_t gray = math::cast<rgba_t>(gmap(i) * scale) & 0xFF;
                                        m_luma(i) = static_cast<luma_t>(gray);
                                }
                        }
                        else
                        {
                                for (coord_t r = 0; r < rows; r ++)
                                {
                                        for (coord_t c = 0; c < cols; c ++)
                                        {
                                                const rgba_t gray = math::cast<rgba_t>(gmap(t + r, l + c) * scale) & 0xFF;
                                                m_luma(r, c) = static_cast<luma_t>(gray);
                                        }
                                }
                        }

                        return setup_luma();
                }

                else if (data.dims() == 3)
                {
                        const auto rmap = data.plane_matrix(0);
                        const auto gmap = data.plane_matrix(1);
                        const auto bmap = data.plane_matrix(2);

                        m_rgba.resize(rows, cols);
                        if (use_all)
                        {
                                for (coord_t i = 0; i < size; i ++)
                                {
                                        const rgba_t red = math::cast<rgba_t>(rmap(i) * scale) & 0xFF;
                                        const rgba_t green = math::cast<rgba_t>(gmap(i) * scale) & 0xFF;
                                        const rgba_t blue = math::cast<rgba_t>(bmap(i) * scale) & 0xFF;
                                        m_rgba(i) = color::make_rgba(red, green, blue);
                                }
                        }
                        else
                        {
                                for (coord_t r = 0; r < rows; r ++)
                                {
                                        for (coord_t c = 0; c < cols; c ++)
                                        {
                                                const rgba_t red = math::cast<rgba_t>(rmap(t + r, l + c) * scale) & 0xFF;
                                                const rgba_t green = math::cast<rgba_t>(gmap(t + r, l + c) * scale) & 0xFF;
                                                const rgba_t blue = math::cast<rgba_t>(bmap(t + r, l + c) * scale) & 0xFF;
                                                m_rgba(r, c) = color::make_rgba(red, green, blue);
                                        }
                                }
                        }

                        return setup_rgba();
                }

                else
                {
                        return false;
                }
        }

        bool image_t::save(const string_t& path) const
        {
                return save_image(path, m_mode, m_rgba, m_luma);
        }

        tensor_t image_t::to_tensor() const
        {
                return to_tensor(geom::make_rect(0, 0, static_cast<coord_t>(cols()), static_cast<coord_t>(rows())));
        }

        tensor_t image_t::to_tensor(const rect_t& region) const
        {
                const coord_t t = geom::top(region), l = geom::left(region);
                const coord_t rows = geom::rows(region), cols = geom::cols(region);
                const coord_t size = rows * cols;

                const coord_t drows = static_cast<coord_t>(m_rows);
                const coord_t dcols = static_cast<coord_t>(m_cols);
                const scalar_t scale = 1.0 / 255.0;

                if (    t < 0 || t + rows >= drows ||
                        l < 0 || l + cols >= dcols)
                {
                        return false;
                }

                const bool use_all = t == 0 && l == 0 && rows == drows && cols == dcols;

                tensor_t data(m_mode == color_mode::luma ? 1 : 3, rows, cols);

                switch (m_mode)
                {
                case color_mode::luma:
                        {
                                auto gmap = data.plane_matrix(0);

                                if (use_all)
                                {
                                        for (coord_t i = 0; i < size; i ++)
                                        {
                                                gmap(i) = scale * m_luma(i);
                                        }
                                }
                                else
                                {
                                        for (coord_t r = 0; r < rows; r ++)
                                        {
                                                for (coord_t c = 0; c < cols; c ++)
                                                {
                                                        gmap(r, c) = scale * m_luma(t + r, l + c);
                                                }
                                        }
                                }
                        }
                        break;

                case color_mode::rgba:
                        {
                                auto rmap = data.plane_matrix(0);
                                auto gmap = data.plane_matrix(1);
                                auto bmap = data.plane_matrix(2);

                                if (use_all)
                                {
                                        for (coord_t i = 0; i < size; i ++)
                                        {
                                                const rgba_t cc = m_rgba(i);
                                                rmap(i) = scale * color::make_red(cc);
                                                gmap(i) = scale * color::make_green(cc);
                                                bmap(i) = scale * color::make_blue(cc);
                                        }
                                }
                                else
                                {
                                        for (coord_t r = 0; r < rows; r ++)
                                        {
                                                for (coord_t c = 0; c < cols; c ++)
                                                {
                                                        const rgba_t cc = m_rgba(t + r, l + c);
                                                        rmap(r, c) = scale * color::make_red(cc);
                                                        gmap(r, c) = scale * color::make_green(cc);
                                                        bmap(r, c) = scale * color::make_blue(cc);
                                                }
                                        }
                                }
                        }
                        break;
                }

                return data;
        }

        bool image_t::make_rgba()
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_rgba.resize(rows(), cols());
                        math::transform(m_luma, m_rgba, [] (luma_t g) { return color::make_rgba(g, g, g); });
                        return setup_rgba();

                case color_mode::rgba:
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::make_luma()
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        return true;

                case color_mode::rgba:
                        m_luma.resize(rows(), cols());
                        math::transform(m_rgba, m_luma, [] (rgba_t c) { return color::make_luma(c); });
                        return setup_luma();

                default:
                        return false;
                }
        }

        bool image_t::fill(rgba_t rgba)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.setConstant(color::make_luma(rgba));
                        return true;

                case color_mode::rgba:
                        m_rgba.setConstant(rgba);
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::fill(luma_t luma)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.setConstant(luma);
                        return true;

                case color_mode::rgba:
                        m_rgba.setConstant(color::make_rgba(luma, luma, luma));
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::copy(coord_t t, coord_t l, const rgba_matrix_t& patch)
        {
                const coord_t rows = static_cast<coord_t>(patch.rows());
                const coord_t cols = static_cast<coord_t>(patch.cols());

                if (    t >= 0 && t + rows < this->rows() &&
                        l >= 0 && l + cols < this->cols())
                {
                        switch (m_mode)
                        {
                        case color_mode::luma:
//                                math::transform(patch, m_luma.block(t, l, rows, cols),
//                                                [] (rgba_t rgba) { return color::make_luma(rgba); });
                                return true;

                        case color_mode::rgba:
                                m_rgba.block(t, l, rows, cols) = patch;
                                return true;

                        default:
                                return false;
                        }
                }

                else
                {
                        return false;
                }
        }

        bool image_t::copy(coord_t t, coord_t l, const luma_matrix_t& patch)
        {

        }

        bool image_t::copy(coord_t t, coord_t l, const image_t& patch)
        {

        }

        bool image_t::copy(coord_t t, coord_t l, const image_t& patch, const rect_t& region)
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
                        m_rows = static_cast<coord_t>(m_luma.rows());
                        m_cols = static_cast<coord_t>(m_luma.cols());
                }

                else if (is_rgba())
                {
                        cielab_matrix_t cielab, cielab_scaled;
                        cielab.resize(rows(), cols());
                        math::transform(m_rgba, cielab, color::make_cielab);

                        math::bilinear(cielab, cielab_scaled, factor);

                        m_rgba.resize(cielab_scaled.rows(), cielab_scaled.cols());
                        math::transform(cielab_scaled, m_rgba, [] (const cielab_t& lab) { return color::make_rgba(lab); });
                        m_rows = static_cast<coord_t>(m_rgba.rows());
                        m_cols = static_cast<coord_t>(m_rgba.cols());
                }
        }
}
