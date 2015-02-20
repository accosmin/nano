#include "image.h"
#include "image_io.h"
#include "libnanocv/util/math.hpp"
#include "libnanocv/util/bilinear.hpp"
#include "libnanocv/util/gaussian.hpp"
#include "libnanocv/util/random_noise.hpp"
#include "libnanocv/util/random_translate.hpp"
#include "libnanocv/tensor/transform.hpp"

namespace ncv
{
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
                m_cols = static_cast<coord_t>(m_rgba.cols());
                return true;
        }

        bool image_t::setup_luma()
        {
                m_rgba.resize(0, 0);
                m_mode = color_mode::luma;
                m_rows = static_cast<coord_t>(m_luma.rows());
                m_cols = static_cast<coord_t>(m_luma.cols());
                return true;
        }

        bool image_t::load_rgba(const string_t& path)
        {
                return  load_rgba_image(path, m_rgba) &&
                        setup_rgba();
        }

        bool image_t::load_luma(const string_t& path)
        {
                return  load_luma_image(path, m_luma) &&
                        setup_luma();
        }
        
        bool image_t::load_rgba(const string_t& name, const char* buffer, size_t buffer_size)
        {
                return  load_rgba_image(name, buffer, buffer_size, m_rgba) &&
                        setup_rgba();
        }
        
        bool image_t::load_luma(const string_t& name, const char* buffer, size_t buffer_size)
        {
                return  load_luma_image(name, buffer, buffer_size, m_luma) &&
                        setup_luma();
        }

        bool image_t::load_luma(const char* buffer, coord_t rows, coord_t cols)
        {
                const coord_t size = rows * cols;

                m_luma.resize(rows, cols);
                tensor::transform(tensor::map_vector(buffer, size),
                                  m_luma, [] (char luma) { return static_cast<luma_t>(luma); });

                return setup_luma();
        }

        bool image_t::load_rgba(const char* buffer, coord_t rows, coord_t cols, coord_t stride)
        {
                const coord_t size = rows * cols;

                m_rgba.resize(rows, cols);
                tensor::transform(tensor::map_vector(buffer + 0 * stride, size),
                                  tensor::map_vector(buffer + 1 * stride, size),
                                  tensor::map_vector(buffer + 2 * stride, size),
                                  m_rgba, [] (char r, char g, char b) { return color::make_rgba(r, g, b); });

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
                tensor::transform(data, m_luma, [] (rgba_t c) { return color::get_luma(c); });

                return setup_luma();
        }

        bool image_t::load_luma(const luma_matrix_t& data)
        {
                m_luma = data;

                return setup_luma();
        }

        bool image_t::load(const tensor_t& data)
        {
                const scalar_t scale = 255;

                if (data.dims() == 1)
                {
                        const auto gmap = data.plane_matrix(0);

                        m_luma.resize(data.rows(), data.cols());
                        tensor::transform(gmap, m_luma, [=] (scalar_t l)
                        {
                                const rgba_t luma = static_cast<rgba_t>(l * scale) & 0xFF;
                                return static_cast<luma_t>(luma);
                        });

                        return setup_luma();
                }

                else if (data.dims() == 3)
                {
                        const auto rmap = data.plane_matrix(0);
                        const auto gmap = data.plane_matrix(1);
                        const auto bmap = data.plane_matrix(2);

                        m_rgba.resize(data.rows(), data.cols());
                        tensor::transform(rmap, gmap, bmap, m_rgba, [=] (scalar_t r, scalar_t g, scalar_t b)
                        {
                                const rgba_t rr = static_cast<rgba_t>(r * scale) & 0xFF;
                                const rgba_t gg = static_cast<rgba_t>(g * scale) & 0xFF;
                                const rgba_t bb = static_cast<rgba_t>(b * scale) & 0xFF;
                                return color::make_rgba(rr, gg, bb);
                        });

                        return setup_rgba();
                }

                else
                {
                        return false;
                }
        }

        bool image_t::save(const string_t& path) const
        {
                switch (m_mode)
                {
                case color_mode::rgba:
                        return save_rgba_image(path, m_rgba);

                case color_mode::luma:
                        return save_luma_image(path, m_luma);

                default:
                        return false;
                }
        }

        tensor_t image_t::to_tensor() const
        {
                return to_tensor(rect_t(0, 0, cols(), rows()));
        }

        tensor_t image_t::to_tensor(const rect_t& region) const
        {
                const coord_t top = region.top();
                const coord_t left = region.left();
                const coord_t rows = region.rows();
                const coord_t cols = region.cols();

                const scalar_t scale = scalar_t(1) / scalar_t(255);

                switch (m_mode)
                {
                case color_mode::luma:
                        {
                                tensor_t data(1, rows, cols);
                                auto gmap = data.plane_matrix(0);

                                tensor::transform(m_luma.block(top, left, rows, cols), gmap, [=] (luma_t luma)
                                {
                                        return scale * luma;
                                });

                                return data;
                        }

                case color_mode::rgba:
                        {
                                tensor_t data(3, rows, cols);
                                auto rmap = data.plane_matrix(0);
                                auto gmap = data.plane_matrix(1);
                                auto bmap = data.plane_matrix(2);

                                tensor::transform(m_rgba.block(top, left, rows, cols), rmap, [=] (rgba_t rgba)
                                {
                                        return scale * color::get_red(rgba);
                                });
                                tensor::transform(m_rgba.block(top, left, rows, cols), gmap, [=] (rgba_t rgba)
                                {
                                        return scale * color::get_green(rgba);
                                });
                                tensor::transform(m_rgba.block(top, left, rows, cols), bmap, [=] (rgba_t rgba)
                                {
                                        return scale * color::get_blue(rgba);
                                });

                                return data;
                        }
                        break;

                default:
                        return tensor_t();
                }
        }

        bool image_t::make_rgba()
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_rgba.resize(rows(), cols());
                        tensor::transform(m_luma, m_rgba, [] (luma_t g) { return color::make_rgba(g, g, g); });

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
                        tensor::transform(m_rgba, m_luma, [] (rgba_t c) { return color::get_luma(c); });

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
                        m_luma.setConstant(color::get_luma(rgba));
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

        bool image_t::fill(const rect_t& rect, rgba_t rgba)
        {
                if (!valid(rect))
                {
                        return false;
                }

                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.block(rect.top(), rect.left(), rect.rows(), rect.cols()).setConstant(color::make_luma(rgba));
                        return true;

                case color_mode::rgba:
                        m_rgba.block(rect.top(), rect.left(), rect.rows(), rect.cols()).setConstant(rgba);
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::fill(const rect_t& rect, luma_t luma)
        {
                if (!valid(rect))
                {
                        return false;
                }

                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.block(rect.top(), rect.left(), rect.rows(), rect.cols()).setConstant(luma);
                        return true;

                case color_mode::rgba:
                        m_rgba.block(rect.top(), rect.left(), rect.rows(), rect.cols()).setConstant(color::make_rgba(luma, luma, luma));
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::copy(coord_t top, coord_t left, const rgba_matrix_t& patch)
        {
                if (!valid(rect_t(left, top, patch.cols(), patch.rows())))
                {
                        return false;
                }

                switch (m_mode)
                {
                case color_mode::luma:
                        {
                                luma_matrix_t luma(patch.rows(), patch.cols());
                                tensor::transform(patch, luma, [] (rgba_t rgba) { return color::make_luma(rgba); });
                                m_luma.block(top, left, patch.rows(), patch.cols()) = luma;
                        }
                        return true;

                case color_mode::rgba:
                        m_rgba.block(top, left, patch.rows(), patch.cols()) = patch;
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::copy(coord_t top, coord_t left, const luma_matrix_t& patch)
        {
                if (!valid(rect_t(left, top, patch.cols(), patch.rows())))
                {
                        return false;
                }

                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.block(top, left, patch.rows(), patch.cols()) = patch;
                        return true;

                case color_mode::rgba:
                        {
                                rgba_matrix_t rgba(patch.rows(), patch.cols());
                                tensor::transform(patch, rgba, [] (luma_t luma) { return color::make_rgba(luma, luma, luma); });
                                m_rgba.block(top, left, patch.rows(), patch.cols()) = rgba;
                        }
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::copy(coord_t top, coord_t left, const image_t& image)
        {
                switch (image.m_mode)
                {
                case color_mode::luma:
                        return copy(top, left, image.m_luma);

                case color_mode::rgba:
                        return copy(top, left, image.m_rgba);

                default:
                        return false;
                }
        }

        bool image_t::copy(coord_t top, coord_t left, const image_t& image, const rect_t& region)
        {
                switch (image.m_mode)
                {
                case color_mode::luma:
                        return  copy(top, left, luma_matrix_t(image.m_luma.block(
                                region.top(), region.left(), region.rows(), region.cols())));

                case color_mode::rgba:
                        return  copy(top, left, rgba_matrix_t(image.m_rgba.block(
                                region.top(), region.left(), region.rows(), region.cols())));

                default:
                        return false;
                }
        }

        bool image_t::set(coord_t row, coord_t col, rgba_t rgba)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma(row, col) = color::make_luma(rgba);
                        return true;

                case color_mode::rgba:
                        m_rgba(row, col) = rgba;
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::set(coord_t row, coord_t col, luma_t luma)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma(row, col) = luma;
                        return true;

                case color_mode::rgba:
                        m_rgba(row, col) = color::make_rgba(luma, luma, luma);
                        return true;

                default:
                        return false;
                }
        }

        void image_t::transpose_in_place()
        {
                m_rgba.transposeInPlace();
                m_luma.transposeInPlace();
        }

        bool image_t::scale(scalar_t factor)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        {
                                luma_matrix_t luma_scaled;
                                ncv::bilinear(m_luma, luma_scaled, factor, color::luma_mixer);
                                m_luma = luma_scaled;
                        }
                        return setup_luma();

                case color_mode::rgba:
                        {
                                rgba_matrix_t rgba_scaled;
                                ncv::bilinear(m_rgba, rgba_scaled, factor, color::rgba_mixer);
                                m_rgba = rgba_scaled;
                        }
                        return setup_rgba();

                default:
                        return false;
                }
        }

        bool image_t::random()
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        m_luma.setRandom();
                        return true;

                case color_mode::rgba:
                        m_rgba.setRandom();
                        tensor::transform(m_rgba, m_rgba, [] (rgba_t c) { return color::set_alpha(c, 255); });
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::random_noise(color_channel channel, scalar_t offset, scalar_t range, scalar_t sigma)
        {
                static const scalar_t cutoff(0.01);
                static const scalar_t cmin(0);
                static const scalar_t cmax(255);

                switch (m_mode)
                {
                case color_mode::luma:
                        return additive_noise(m_luma, offset, range, sigma, cutoff, cmin, cmax, color::get_luma, color::set_luma);

                case color_mode::rgba:
                        switch (channel)
                        {
                        case color_channel::red:
                                return additive_noise(m_rgba, offset, range, sigma, cutoff, cmin, cmax, color::get_red, color::set_red);

                        case color_channel::green:
                                return additive_noise(m_rgba, offset, range, sigma, cutoff, cmin, cmax, color::get_green, color::set_green);

                        case color_channel::blue:
                                return additive_noise(m_rgba, offset, range, sigma, cutoff, cmin, cmax, color::get_blue, color::set_blue);

                        default:
                                return additive_noise(m_rgba, offset, range, sigma, cutoff, cmin, cmax, color::get_red, color::set_red) &&
                                       additive_noise(m_rgba, offset, range, sigma, cutoff, cmin, cmax, color::get_green, color::set_green) &&
                                       additive_noise(m_rgba, offset, range, sigma, cutoff, cmin, cmax, color::get_blue, color::set_blue);
                        }

                default:
                        return false;
                }
        }

        bool image_t::random_translate(coord_t range)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        return ncv::random_translate(m_luma, range, color::luma_mixer);

                case color_mode::rgba:
                        return ncv::random_translate(m_rgba, range, color::rgba_mixer);

                default:
                        return false;
                }
        }

        bool image_t::gauss(color_channel channel, scalar_t sigma)
        {
                static const scalar_t cutoff(0.01);
                static const scalar_t cmin(0);
                static const scalar_t cmax(255);

                switch (m_mode)
                {
                case color_mode::luma:
                        return gaussian(m_luma, sigma, cutoff, cmin, cmax, color::get_luma, color::set_luma);

                case color_mode::rgba:
                        switch (channel)
                        {
                        case color_channel::red:
                                return gaussian(m_rgba, sigma, cutoff, cmin, cmax, color::get_red, color::set_red);

                        case color_channel::green:
                                return gaussian(m_rgba, sigma, cutoff, cmin, cmax, color::get_green, color::set_green);

                        case color_channel::blue:
                                return gaussian(m_rgba, sigma, cutoff, cmin, cmax, color::get_blue, color::set_blue);

                        default:
                                return gaussian(m_rgba, sigma, cutoff, cmin, cmax, color::get_red, color::set_red) &&
                                       gaussian(m_rgba, sigma, cutoff, cmin, cmax, color::get_green, color::set_green) &&
                                       gaussian(m_rgba, sigma, cutoff, cmin, cmax, color::get_blue, color::set_blue);
                        }

                default:
                        return false;
                }
        }

        bool image_t::alpha_blend(const rgba_matrix_t& patch)
        {
                if (    patch.rows() != rows() ||
                        patch.cols() != cols())
                {
                        return false;
                }

                switch (m_mode)
                {
                case color_mode::luma:
                        for (coord_t i = 0; i < size(); i ++)
                        {
                                const rgba_t rgba1 = patch(i);

                                const rgba_t alpha1 = color::get_alpha(rgba1);
                                const rgba_t alpha2 = 255;

                                m_luma(i) = (alpha1 * color::get_luma(rgba1) + alpha2 * m_luma(i)) / (alpha1 + alpha2);
                        }
                        return true;

                case color_mode::rgba:
                        for (coord_t i = 0; i < size(); i ++)
                        {
                                const rgba_t rgba1 = patch(i);
                                const rgba_t rgba2 = m_rgba(i);

                                const rgba_t alpha1 = color::get_alpha(rgba1);
                                const rgba_t alpha2 = color::get_alpha(rgba2);
                                const rgba_t alphax = alpha1 + alpha2;

                                m_rgba(i) = color::make_rgba(
                                        (alpha1 * color::get_red(rgba1) + alpha2 * color::get_red(rgba2)) / alphax,
                                        (alpha1 * color::get_green(rgba1) + alpha2 * color::get_green(rgba2)) / alphax,
                                        (alpha1 * color::get_blue(rgba1) + alpha2 * color::get_blue(rgba2)) / alphax,
                                        std::max(alpha1, alpha2));
                        }
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::fill_rectangle(const rect_t& rect, rgba_t rgba)
        {
                return fill(rect, rgba);
        }

        namespace
        {
                template
                <
                        typename tmatrix,
                        typename tvalue
                >
                bool setup_circle(const rect_t& rect, tmatrix& data, tvalue fill_value)
                {
                        const point_t center = rect.center();
                        const coord_t cx = center.x();
                        const coord_t cy = center.y();

                        const coord_t radius = (std::min(rect.width(), rect.height()) + 1) / 2;
                        const coord_t radius2 = radius * radius;

                        const coord_t l = std::max(rect.left(), coord_t(0));
                        const coord_t r = std::min(rect.right(), static_cast<coord_t>(data.cols()));
                        const coord_t t = std::max(rect.top(), coord_t(0));
                        const coord_t b = std::min(rect.bottom(), static_cast<coord_t>(data.rows()));

                        for (coord_t x = l; x < r; x ++)
                        {
                                for (coord_t y = t; y < b; y ++)
                                {
                                        if (math::square(x - cx) + math::square(y - cy) < radius2)
                                        {
                                                data(y, x) = fill_value;
                                        }
                                }
                        }

                        return true;
                }
        }

        bool image_t::fill_circle(const rect_t& rect, rgba_t rgba)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        return setup_circle(rect, m_luma, color::get_luma(rgba));

                case color_mode::rgba:
                        return setup_circle(rect, m_rgba, rgba);

                default:
                        return false;
                }
        }

        namespace
        {
                template
                <
                        typename tmatrix,
                        typename tvalue
                >
                bool setup_ellipse(const rect_t& rect, tmatrix& data, tvalue fill_value)
                {
                        const point_t center = rect.center();
                        const coord_t cx = center.x();
                        const coord_t cy = center.y();

                        const coord_t radiusx = (rect.width() + 1) / 2;
                        const coord_t radiusy = (rect.height() + 1) / 2;

                        const coord_t radiusx2 = radiusx * radiusx;
                        const coord_t radiusy2 = radiusy * radiusy;

                        const coord_t radius2 = radiusx2 * radiusy2;

                        const coord_t l = std::max(rect.left(), coord_t(0));
                        const coord_t r = std::min(rect.right(), static_cast<coord_t>(data.cols()));
                        const coord_t t = std::max(rect.top(), coord_t(0));
                        const coord_t b = std::min(rect.bottom(), static_cast<coord_t>(data.rows()));

                        for (coord_t x = l; x < r; x ++)
                        {
                                for (coord_t y = t; y < b; y ++)
                                {
                                        if (math::square(x - cx) * radiusy2 + math::square(y - cy) * radiusx2 < radius2)
                                        {
                                                data(y, x) = fill_value;
                                        }
                                }
                        }

                        return true;
                }
        }

        bool image_t::fill_ellipse(const rect_t& rect, rgba_t rgba)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        return setup_ellipse(rect, m_luma, color::get_luma(rgba));

                case color_mode::rgba:
                        return setup_ellipse(rect, m_rgba, rgba);

                default:
                        return false;
                }
        }

        namespace
        {
                template
                <
                        typename tmatrix,
                        typename tvalue
                >
                bool setup_up_triangle(const rect_t& rect, tmatrix& data, tvalue fill_value)
                {
                        const coord_t w = rect.width(), w2 = (w + 1) / 2;
                        const coord_t h = rect.height();

                        const coord_t l = std::max(rect.left(), coord_t(0));
                        const coord_t r = std::min(rect.right(), static_cast<coord_t>(data.cols()));
                        const coord_t t = std::max(rect.top(), coord_t(0));
                        const coord_t b = std::min(rect.bottom(), static_cast<coord_t>(data.rows()));

                        for (coord_t x = l; x < r; x ++)
                        {
                                const coord_t dy = (h * math::abs(x - l - w2) + w2 - 1) / w2;

                                for (coord_t y = t; y < b; y ++)
                                {
                                        if (y - t >= dy)
                                        {
                                                data(y, x) = fill_value;
                                        }
                                }
                        }

                        return true;
                }

                template
                <
                        typename tmatrix,
                        typename tvalue
                >
                bool setup_down_triangle(const rect_t& rect, tmatrix& data, tvalue fill_value)
                {
                        const coord_t w = rect.width(), w2 = (w + 1) / 2;
                        const coord_t h = rect.height();

                        const coord_t l = std::max(rect.left(), coord_t(0));
                        const coord_t r = std::min(rect.right(), static_cast<coord_t>(data.cols()));
                        const coord_t t = std::max(rect.top(), coord_t(0));
                        const coord_t b = std::min(rect.bottom(), static_cast<coord_t>(data.rows()));

                        for (coord_t x = l; x < r; x ++)
                        {
                                const coord_t dy = (h * (x < l + w2 ? x - l : r - x) + w2 - 1) / w2;

                                for (coord_t y = t; y < b; y ++)
                                {
                                        if (y - t <= dy)
                                        {
                                                data(y, x) = fill_value;
                                        }
                                }
                        }

                        return true;
                }
        }

        bool image_t::fill_up_triangle(const rect_t& rect, rgba_t rgba)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        return setup_up_triangle(rect, m_luma, color::get_luma(rgba));

                case color_mode::rgba:
                        return setup_up_triangle(rect, m_rgba, rgba);

                default:
                        return false;
                }
        }

        bool image_t::fill_down_triangle(const rect_t& rect, rgba_t rgba)
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        return setup_down_triangle(rect, m_luma, color::get_luma(rgba));

                case color_mode::rgba:
                        return setup_down_triangle(rect, m_rgba, rgba);

                default:
                        return false;
                }
        }
}
