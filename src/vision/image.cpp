#include "image.h"
#include "image_io.h"
#include "math/cast.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "tensor/transform.hpp"

namespace nano
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
                                  m_rgba, [] (char r, char g, char b)
                {
                        return color::make_rgba(static_cast<unsigned char>(r),
                                                static_cast<unsigned char>(g),
                                                static_cast<unsigned char>(b));
                });

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
                tensor::transform(data, m_luma, [] (rgba_t c) { return color::make_luma(c); });

                return setup_luma();
        }

        bool image_t::load_luma(const luma_matrix_t& data)
        {
                m_luma = data;

                return setup_luma();
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

                switch (m_mode)
                {
                case color_mode::luma:
                        return color::to_luma_tensor(luma_matrix_t(m_luma.block(top, left, rows, cols)));

                case color_mode::rgba:
                        return color::to_rgb_tensor(rgba_matrix_t(m_rgba.block(top, left, rows, cols)));

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
                        tensor::transform(m_rgba, m_luma, [] (rgba_t c) { return color::make_luma(c); });

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

        bool image_t::random()
        {
                switch (m_mode)
                {
                case color_mode::luma:
                        tensor::set_random(m_luma, nano::random_t<luma_t>());
                        return true;

                case color_mode::rgba:
                        tensor::set_random(m_rgba, nano::random_t<rgba_t>());
                        tensor::transform(m_rgba, m_rgba, [] (rgba_t c) { return color::set_alpha(c, 255); });
                        return true;

                default:
                        return false;
                }
        }
}
