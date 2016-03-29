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
                                {
                                        auto band0 = image.vector(0);
                                        for (int i = 0; i < size; ++ i, data += 4)
                                        {
                                                band0(i) = static_cast<luma_t>(make_luma(data[0], data[1], data[2]));
                                        }
                                }
                                break;

                        case color_mode::rgba:
                                image.resize(4, rows, cols);
                                {
                                        auto band0 = image.vector(0);
                                        auto band1 = image.vector(1);
                                        auto band2 = image.vector(2);
                                        auto band3 = image.vector(3);
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
                                        auto band0 = image.vector(0);
                                        auto band1 = image.vector(1);
                                        auto band2 = image.vector(2);
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

                                tensor::vector_t<luma_t> temp(rows * cols);
                                for (auto r = 0, i = 0; r < rows; ++ r)
                                {
                                        for (auto c = 0; c < cols; ++ c, ++ i)
                                        {
                                                temp(i) = band0(rows - 1 - r, c);
                                        }
                                }
                                ret = ilTexImage(static_cast<ILuint>(cols), static_cast<ILuint>(rows),
                                                 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, (void*)temp.data());
                        }
                        break;

                case 3:
                        {
                                const auto band0 = image.matrix(0);
                                const auto band1 = image.matrix(1);
                                const auto band2 = image.matrix(2);

                                tensor::vector_t<luma_t> temp(rows * cols * 3);
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
                                                 1, 3, IL_RGB, IL_UNSIGNED_BYTE, (void*)temp.data());
                        }
                        break;

                case 4:
                        {
                                const auto band0 = image.matrix(0);
                                const auto band1 = image.matrix(1);
                                const auto band2 = image.matrix(2);
                                const auto band3 = image.matrix(3);

                                tensor::vector_t<luma_t> temp(rows * cols * 4);
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

        bool load_rgba_image(const string_t& path, image_tensor_t& image)
        {
                return load_image(path, color_mode::rgba, image);
        }

        bool load_luma_image(const string_t& path, image_tensor_t& image)
        {
                return load_image(path, color_mode::luma, image);
        }

        bool load_rgb_image(const string_t& path, image_tensor_t& image)
        {
                return load_image(path, color_mode::rgb, image);
        }

        bool load_rgba_image(const string_t& name, const char* buffer, const size_t buffer_size, image_tensor_t& image)
        {
                return load_image(name, buffer, buffer_size, color_mode::rgba, image);
        }

        bool load_luma_image(const string_t& name, const char* buffer, const size_t buffer_size, image_tensor_t& image)
        {
                return load_image(name, buffer, buffer_size, color_mode::luma, image);
        }

        bool load_rgb_image(const string_t& name, const char* buffer, const size_t buffer_size, image_tensor_t& image)
        {
                return load_image(name, buffer, buffer_size, color_mode::rgb, image);
        }
}
