#include "image_io.h"
#include "text/algorithm.h"
#include <map>
#include <IL/il.h>

namespace nano
{
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
                                for (int i = 0; i < size; ++ i, data += 4)
                                {
                                        image.vector(0)(i) = color::make_luma(data[0], data[1], data[2]);
                                }
                                break;

                        case color_mode::rgba:
                                image.resize(4, rows, cols);
                                for (int i = 0; i < size; ++ i, data += 4)
                                {
                                        image.vector(0)(i) = data[0];
                                        image.vector(1)(i) = data[1];
                                        image.vector(2)(i) = data[2];
                                        image.vector(3)(i) = data[3];
                                }
                                break;

                        case color_mode::rgb:
                                image.resize(3, rows, cols);
                                for (int i = 0; i < size; ++ i, data += 4)
                                {
                                        image.vector(0)(i) = data[0];
                                        image.vector(1)(i) = data[1];
                                        image.vector(2)(i) = data[2];
                                }
                                break;
                        }

                        ret = true;
                }

                return ret;
        }

        static bool load_image(const string_t& path, const color_mode mode, image_tensor_& image)
        {
                ilInit();

                const ILuint id = ilGenImage();
                ilBindImage(id);

                const bool ret =
                        ilLoadImage((const ILstring)path.c_str()) &&
                        load_image(mode, image);

                ilDeleteImage(id);

                return ret;
        }

        static bool load_image(const string_t& name, const char* buffer, const size_t buffer_size,
                const color_mode mode, image_tensor_t& image)
        {
                ilInit();

                const ILuint id = ilGenImage();
                ilBindImage(id);

                const std::map<string_t, ILenum> extensions =
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

                ILenum type = IL_TYPE_UNKNOWN;
                for (const auto& extension : extensions)
                {
                        if (nano::iends_with(name, extension.first))
                        {
                                type = extension.second;
                        }
                }

                const bool ret =
                        ilLoadL(type, buffer, static_cast<unsigned int>(buffer_size)) &&
                        load_image(mode, image);

                ilDeleteImage(id);

                return ret;
        }

        bool save_image(const string_t& path, const image_tensor_t& image)
        {
                const auto rows = mode == color_mode::rgba ? rgba.rows() : luma.rows();
                const auto cols = mode == color_mode::rgba ? rgba.cols() : luma.cols();

                ilInit();

                const ILuint id = ilGenImage();
                ilBindImage(id);

                bool ret = true;

                switch (mode)
                {
                case color_mode::luma:
                        {
                                luma_matrix_t temp(rows, cols);
                                for (auto r = 0; r < rows; ++ r)
                                {
                                        for (auto c = 0; c < cols; ++ c)
                                        {
                                                const luma_t val = luma(rows - 1 - r, c);

                                                temp(r, c) = val;
                                        }
                                }
                                ret = ilTexImage(static_cast<ILuint>(cols), static_cast<ILuint>(rows),
                                                 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, (void*)temp.data());
                        }
                        break;

                case color_mode::rgba:
                        {
                                rgba_matrix_t temp(rows, cols);
                                for (auto r = 0; r < rows; ++ r)
                                {
                                        for (auto c = 0; c < cols; ++ c)
                                        {
                                                const rgba_t val = rgba(rows - 1 - r, c);
                                                const rgba_t cr = color::get_red(val);
                                                const rgba_t cg = color::get_green(val);
                                                const rgba_t cb = color::get_blue(val);
                                                const rgba_t ca = color::get_alpha(val);

                                                temp(r, c) = color::make_rgba(ca, cb, cg, cr);
                                        }
                                }
                                ret = ilTexImage(static_cast<ILuint>(cols), static_cast<ILuint>(rows),
                                                 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, (void*)temp.data());
                        }
                        break;
                }

                ret =   ret &&
                        ilEnable(IL_FILE_OVERWRITE) &&
                        ilSaveImage((const ILstring)path.c_str());

                ilDeleteImage(id);

                return ret;
        }

        bool load_rgba_image(const string_t& path, rgba_matrix_t& rgba)
        {
                luma_matrix_t luma;
                return load_image(path, color_mode::rgba, rgba, luma);
        }

        bool load_rgba_image(const string_t& name, const char* buffer, size_t buffer_size, rgba_matrix_t& rgba)
        {
                luma_matrix_t luma;
                return load_image(name, buffer, buffer_size, color_mode::rgba, rgba, luma);
        }

        bool load_luma_image(const string_t& path, luma_matrix_t& luma)
        {
                rgba_matrix_t rgba;
                return load_image(path, color_mode::luma, rgba, luma);
        }

        bool load_luma_image(const string_t& name, const char* buffer, size_t buffer_size, luma_matrix_t& luma)
        {
                rgba_matrix_t rgba;
                return load_image(name, buffer, buffer_size, color_mode::luma, rgba, luma);
        }
}
