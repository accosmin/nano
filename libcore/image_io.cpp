#include "image_io.h"
#include "libtext/ends_with.hpp"
#include <map>
#include <IL/il.h>

namespace ncv
{
        static bool load_image(color_mode mode, rgba_matrix_t& rgba, luma_matrix_t& luma)
        {
                bool ret = false;

                if (ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE))
                {
                        const ILint cols = ilGetInteger(IL_IMAGE_WIDTH);
                        const ILint rows = ilGetInteger(IL_IMAGE_HEIGHT);
                        const ILubyte* data = ilGetData();

                        switch (mode)
                        {
                        case color_mode::luma:
                                luma.resize(rows, cols);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                const ILubyte* pix = data + 4 * (r * cols + c);
                                                luma(r, c) = color::make_luma(pix[0], pix[1], pix[2]);
                                        }
                                }
                                break;

                        case color_mode::rgba:
                                rgba.resize(rows, cols);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                const ILubyte* pix = data + 4 * (r * cols + c);
                                                rgba(r, c) = color::make_rgba(pix[0], pix[1], pix[2], pix[3]);
                                        }
                                }
                                break;
                        }
                        
                        ret = true;
                }
                
                return ret;
        }
        
        static bool load_image(const string_t& path, 
                color_mode mode, rgba_matrix_t& rgba, luma_matrix_t& luma)
        {
                ilInit();
                
                const ILuint id = ilGenImage();
                ilBindImage(id);
                
                const bool ret = 
                        ilLoadImage((const ILstring)path.c_str()) &&
                        load_image(mode, rgba, luma);
                
                ilDeleteImage(id);
                
                return ret;
        }
        
        static bool load_image(const string_t& name, const char* buffer, size_t buffer_size,
                color_mode mode, rgba_matrix_t& rgba, luma_matrix_t& luma)
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
                        if (text::iends_with(name, extension.first))
                        {
                                type = extension.second;
                        }
                }

                const bool ret = 
                        ilLoadL(type, buffer, buffer_size) &&
                        load_image(mode, rgba, luma);

                ilDeleteImage(id);

                return ret;
        }

        static bool save_image(const string_t& path,
                color_mode mode, const rgba_matrix_t& rgba, const luma_matrix_t& luma)
        {
                const int rows = static_cast<int>(mode == color_mode::rgba ? rgba.rows() : luma.rows());
                const int cols = static_cast<int>(mode == color_mode::rgba ? rgba.cols() : luma.cols());

                ilInit();

                const ILuint id = ilGenImage();
                ilBindImage(id);

                bool ret = true;

                switch (mode)
                {
                case color_mode::luma:
                        {
                                luma_matrix_t temp(rows, cols);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                const luma_t val = luma(rows - 1 - r, c);

                                                temp(r, c) = val;
                                        }
                                }
                                ret = ilTexImage(cols, rows, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, (void*)temp.data());
                        }
                        break;

                case color_mode::rgba:
                        {
                                rgba_matrix_t temp(rows, cols);
                                for (int r = 0; r < rows; r ++)
                                {
                                        for (int c = 0; c < cols; c ++)
                                        {
                                                const rgba_t val = rgba(rows - 1 - r, c);
                                                const rgba_t cr = color::get_red(val);
                                                const rgba_t cg = color::get_green(val);
                                                const rgba_t cb = color::get_blue(val);
                                                const rgba_t ca = color::get_alpha(val);

                                                temp(r, c) = color::make_rgba(ca, cb, cg, cr);
                                        }
                                }
                                ret = ilTexImage(cols, rows, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, (void*)temp.data());
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

        bool save_rgba_image(const string_t& path, const rgba_matrix_t& rgba)
        {
                luma_matrix_t luma;
                return save_image(path, color_mode::rgba, rgba, luma);
        }

        bool save_luma_image(const string_t& path, const luma_matrix_t& luma)
        {
                rgba_matrix_t rgba;
                return save_image(path, color_mode::luma, rgba, luma);
        }
}
