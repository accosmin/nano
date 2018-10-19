#include <il.h>
#include "hash.h"
#include "image.h"
#include "algorithm.h"

using namespace nano;

static bool load_image(const color_mode mode, image_tensor_t& image)
{
        bool ret = false;

        if (ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE))
        {
                const ILint cols = ilGetInteger(IL_IMAGE_WIDTH);
                const ILint rows = ilGetInteger(IL_IMAGE_HEIGHT);
                const ILint size = rows * cols;
                const ILubyte* data = ilGetData();

                switch (mode)
                {
                case color_mode::luma:
                        image.resize(1, rows, cols);
                        {
                                auto band0 = image.matrix(0);
                                for (int i = 0; i < size; ++ i, data += 4)
                                {
                                        band0(i) = static_cast<luma_t>(make_luma(data[0], data[1], data[2]));
                                }
                        }
                        break;

                case color_mode::rgba:
                        image.resize(4, rows, cols);
                        {
                                auto band0 = image.matrix(0);
                                auto band1 = image.matrix(1);
                                auto band2 = image.matrix(2);
                                auto band3 = image.matrix(3);
                                for (int i = 0; i < size; ++ i, data += 4)
                                {
                                        band0(i) = data[0];
                                        band1(i) = data[1];
                                        band2(i) = data[2];
                                        band3(i) = data[3];
                                }
                        }
                        break;

                case color_mode::rgb:
                        image.resize(3, rows, cols);
                        {
                                auto band0 = image.matrix(0);
                                auto band1 = image.matrix(1);
                                auto band2 = image.matrix(2);
                                for (int i = 0; i < size; ++ i, data += 4)
                                {
                                        band0(i) = data[0];
                                        band1(i) = data[1];
                                        band2(i) = data[2];
                                }
                        }
                        break;
                }

                ret = true;
        }

        return ret;
}

static bool load_image(const string_t& path, const color_mode mode, image_tensor_t& image)
{
        ilInit();

        const ILuint id = ilGenImage();
        ilBindImage(id);

        const bool ret =
                ilLoadImage(static_cast<const ILstring>(path.c_str())) &&
                load_image(mode, image);

        ilDeleteImage(id);

        return ret;
}

static ILenum get_type(const string_t& name)
{
        const std::vector<std::pair<const char*, ILenum>> extensions =
        {
                { ".pgm",       IL_PNM },
                { ".ppm",       IL_PNM },
                { ".png",       IL_PNG },
                { ".tif",       IL_TIF },
                { ".tiff",      IL_TIF },
                { ".jpeg",      IL_JPG },
                { ".jpg",       IL_JPG },
                { ".bmp",       IL_BMP },
        };

        for (const auto& extension : extensions)
        {
                if (nano::iends_with(name, extension.first))
                {
                        return extension.second;
                }
        }
        return IL_TYPE_UNKNOWN;
}

static bool load_image(const string_t& name, const char* buffer, const size_t buffer_size,
        const color_mode mode, image_tensor_t& image)
{
        ilInit();

        const auto id = ilGenImage();
        ilBindImage(id);

        const auto type = get_type(name);
        const auto code =
                ilLoadL(type, buffer, static_cast<unsigned int>(buffer_size)) &&
                load_image(mode, image);

        ilDeleteImage(id);

        return code;
}

static bool save_image(const string_t& path, const image_tensor_t& image)
{
        const auto rows = image.rows();
        const auto cols = image.cols();

        ilInit();

        const ILuint id = ilGenImage();
        ilBindImage(id);

        bool ret = true;

        switch (image.size<0>())
        {
        case 1:
                {
                        const auto band0 = image.matrix(0);

                        tensor_vector_t<luma_t> temp(rows * cols);
                        for (auto r = 0, i = 0; r < rows; ++ r)
                        {
                                for (auto c = 0; c < cols; ++ c, ++ i)
                                {
                                        temp(i) = band0(rows - 1 - r, c);
                                }
                        }
                        ret = ilTexImage(static_cast<ILuint>(cols), static_cast<ILuint>(rows),
                                         1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, reinterpret_cast<void*>(temp.data()));
                }
                break;

        case 3:
                {
                        const auto band0 = image.matrix(0);
                        const auto band1 = image.matrix(1);
                        const auto band2 = image.matrix(2);

                        tensor_vector_t<luma_t> temp(rows * cols * 3);
                        for (auto r = 0, i = 0; r < rows; ++ r)
                        {
                                for (auto c = 0; c < cols; ++ c, i += 3)
                                {
                                        temp(i + 0) = band0(rows - 1 - r, c);
                                        temp(i + 1) = band1(rows - 1 - r, c);
                                        temp(i + 2) = band2(rows - 1 - r, c);
                                }
                        }
                        ret = ilTexImage(static_cast<ILuint>(cols), static_cast<ILuint>(rows),
                                         1, 3, IL_RGB, IL_UNSIGNED_BYTE, reinterpret_cast<void*>(temp.data()));
                }
                break;

        case 4:
                {
                        const auto band0 = image.matrix(0);
                        const auto band1 = image.matrix(1);
                        const auto band2 = image.matrix(2);
                        const auto band3 = image.matrix(3);

                        tensor_vector_t<luma_t> temp(rows * cols * 4);
                        for (auto r = 0, i = 0; r < rows; ++ r)
                        {
                                for (auto c = 0; c < cols; ++ c, i += 4)
                                {
                                        temp(i + 0) = band0(rows - 1 - r, c);
                                        temp(i + 1) = band1(rows - 1 - r, c);
                                        temp(i + 2) = band2(rows - 1 - r, c);
                                        temp(i + 3) = band3(rows - 1 - r, c);
                                }
                        }
                        ret = ilTexImage(static_cast<ILuint>(cols), static_cast<ILuint>(rows),
                                         1, 4, IL_RGBA, IL_UNSIGNED_BYTE, reinterpret_cast<void*>(temp.data()));
                }
                break;

        }

        ret =   ret &&
                ilEnable(IL_FILE_OVERWRITE) &&
                ilSaveImage(static_cast<const ILstring>(path.c_str()));

        ilDeleteImage(id);

        return ret;
}

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
        return load_image(path, color_mode::rgba, m_data);
}

bool image_t::load_luma(const string_t& path)
{
        return load_image(path, color_mode::luma, m_data);
}

bool image_t::load_rgb(const string_t& path)
{
        return load_image(path, color_mode::rgb, m_data);
}

bool image_t::load_rgba(const string_t& name, const char* buffer, const size_t buffer_size)
{
        return load_image(name, buffer, buffer_size, color_mode::rgba, m_data);
}

bool image_t::load_luma(const string_t& name, const char* buffer, const size_t buffer_size)
{
        return load_image(name, buffer, buffer_size, color_mode::luma, m_data);
}

bool image_t::load_rgb(const string_t& name, const char* buffer, const size_t buffer_size)
{
        return load_image(name, buffer, buffer_size, color_mode::rgb, m_data);
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
