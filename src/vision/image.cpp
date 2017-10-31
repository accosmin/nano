#include "image.h"
#include "image_io.h"
#include "math/hash.h"

using namespace nano;

static tensor_size_t mode_to_dims(const color_mode mode)
{
        switch (mode)
        {
        case color_mode::luma:  return 1;
        case color_mode::rgb:   return 3;
        case color_mode::rgba:  return 4;
        default:                return 0;
        }
}

image_t::image_t(const coord_t rows, const coord_t cols, const color_mode mode) :
        m_data(mode_to_dims(mode), rows, cols)
{
}

size_t image_t::hash() const
{
        return nano::hash_range(m_data.data(), m_data.data() + m_data.size());
}

void image_t::resize(const coord_t rows, const coord_t cols, const color_mode mode)
{
        m_data.resize(mode_to_dims(mode), rows, cols);
}

bool image_t::load_rgba(const string_t& path)
{
        return load_rgba_image(path, m_data);
}

bool image_t::load_luma(const string_t& path)
{
        return load_luma_image(path, m_data);
}

bool image_t::load_rgb(const string_t& path)
{
        return load_rgb_image(path, m_data);
}

bool image_t::load_rgba(const string_t& name, const char* buffer, const size_t buffer_size)
{
        return load_rgba_image(name, buffer, buffer_size, m_data);
}

bool image_t::load_luma(const string_t& name, const char* buffer, const size_t buffer_size)
{
        return load_luma_image(name, buffer, buffer_size, m_data);
}

bool image_t::load_rgb(const string_t& name, const char* buffer, const size_t buffer_size)
{
        return load_rgb_image(name, buffer, buffer_size, m_data);
}

bool image_t::load_luma(const char* buffer, const coord_t rows, const coord_t cols)
{
        resize(rows, cols, color_mode::luma);
        plane(0) = nano::map_matrix(buffer, rows, cols).cast<luma_t>();
        return true;
}

bool image_t::load_rgba(const char* buffer, const coord_t rows, const coord_t cols, const coord_t stride)
{
        resize(rows, cols, color_mode::rgba);
        plane(0) = nano::map_matrix(buffer + 0 * stride, rows, cols).cast<luma_t>();
        plane(1) = nano::map_matrix(buffer + 1 * stride, rows, cols).cast<luma_t>();
        plane(2) = nano::map_matrix(buffer + 2 * stride, rows, cols).cast<luma_t>();
        plane(3) = nano::map_matrix(buffer + 3 * stride, rows, cols).cast<luma_t>();
        return true;
}

bool image_t::load_rgb(const char* buffer, const coord_t rows, const coord_t cols, const coord_t stride)
{
        resize(rows, cols, color_mode::rgb);
        plane(0) = nano::map_matrix(buffer + 0 * stride, rows, cols).cast<luma_t>();
        plane(1) = nano::map_matrix(buffer + 1 * stride, rows, cols).cast<luma_t>();
        plane(2) = nano::map_matrix(buffer + 2 * stride, rows, cols).cast<luma_t>();
        return true;
}

bool image_t::load(const image_tensor_t& data)
{
        m_data = data;
        return true;
}

bool image_t::save(const string_t& path) const
{
        return save_image(path, m_data);
}

tensor3d_t image_t::to_tensor() const
{
        tensor3d_t ret(dims(), rows(), cols());
        ret.vector() = m_data.vector().cast<scalar_t>() * static_cast<scalar_t>(1.0 / 255.0);
        return ret;
}

tensor3d_t image_t::to_tensor(const rect_t& rect) const
{
        tensor3d_t ret(dims(), rect.height(), rect.width());
        for (auto i = 0; i < dims(); ++ i)
        {
                ret.matrix(i) = plane(i, rect).cast<scalar_t>() * static_cast<scalar_t>(1.0 / 255.0);
        }
        return ret;
}

bool image_t::from_tensor(const tensor3d_t& data)
{
        switch (data.size<0>())
        {
        case 1:
        case 3:
        case 4:
                m_data.resize(data.size<0>(), data.rows(), data.cols());
                m_data.vector() = (data.array() * 255).max(0).min(255).cast<luma_t>();
                return true;

        default:
                return false;
        }
}

void image_t::make_rgba()
{
        switch (mode())
        {
        case color_mode::luma:
                {
                        const auto data = m_data;
                        resize(rows(), cols(), color_mode::rgba);
                        m_data.matrix(0) = data.matrix(0);
                        m_data.matrix(1) = data.matrix(0);
                        m_data.matrix(2) = data.matrix(0);
                        m_data.matrix(3).setConstant(255);      // no alpha!
                }
                break;

        case color_mode::rgb:
                {
                        const auto data = m_data;
                        resize(rows(), cols(), color_mode::rgba);
                        m_data.matrix(0) = data.matrix(0);
                        m_data.matrix(1) = data.matrix(1);
                        m_data.matrix(2) = data.matrix(2);
                        m_data.matrix(3).setConstant(255);      // no alpha!
                }
                break;

        default:
                break;
        }
}

void image_t::make_rgb()
{
        switch (mode())
        {
        case color_mode::luma:
                {
                        const auto data = m_data;
                        resize(rows(), cols(), color_mode::rgb);
                        m_data.matrix(0) = data.matrix(0);
                        m_data.matrix(1) = data.matrix(0);
                        m_data.matrix(2) = data.matrix(0);
                }
                break;

        case color_mode::rgba:
                {
                        const auto data = m_data;
                        resize(rows(), cols(), color_mode::rgb);
                        m_data.matrix(0) = data.matrix(0);
                        m_data.matrix(1) = data.matrix(1);
                        m_data.matrix(2) = data.matrix(2);
                }
                break;

        default:
                break;
        }
}

void image_t::make_luma()
{
        switch (mode())
        {
        case color_mode::rgb:
        case color_mode::rgba:
                {
                        const auto data = m_data;
                        resize(rows(), cols(), color_mode::luma);
                        plane(0) = nano::make_luma(data.matrix(0), data.matrix(1), data.matrix(2)).cast<luma_t>();
                }
                break;

        default:
                break;
        }
}

void image_t::fill(const rgba_t& rgba)
{
        switch (mode())
        {
        case color_mode::luma:
                plane(0).setConstant(static_cast<luma_t>(nano::make_luma(rgba(0), rgba(1), rgba(2))));
                break;

        case color_mode::rgb:
                plane(0).setConstant(rgba(0));
                plane(1).setConstant(rgba(1));
                plane(2).setConstant(rgba(2));
                break;

        case color_mode::rgba:
                plane(0).setConstant(rgba(0));
                plane(1).setConstant(rgba(1));
                plane(2).setConstant(rgba(2));
                plane(3).setConstant(rgba(3));
                break;
        }
}

void image_t::fill(const rgb_t& rgb)
{
        return fill(rgba_t{rgb(0), rgb(1), rgb(2), 255});
}

void image_t::fill(const luma_t luma)
{
        return fill(rgba_t{luma, luma, luma, 255});
}

color_mode image_t::mode() const
{
        switch (dims())
        {
        case 1:         return color_mode::luma;
        case 3:         return color_mode::rgb;
        case 4:         return color_mode::rgba;
        default:        return color_mode::luma;
        }
}
