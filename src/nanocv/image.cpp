#include "image.h"
#include "util/bilinear.hpp"
#include "tensor/transform.hpp"
#include <IL/il.h>
#include <map>

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
                        if (boost::algorithm::iends_with(name, extension.first))
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
                                                const rgba_t cr = color::make_red(val);
                                                const rgba_t cg = color::make_green(val);
                                                const rgba_t cb = color::make_blue(val);
                                                const rgba_t ca = color::make_alpha(val);

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
                return  load_image(path, color_mode::rgba, m_rgba, m_luma) &&
                        setup_rgba();
        }

        bool image_t::load_luma(const string_t& path)
        {
                return  load_image(path, color_mode::luma, m_rgba, m_luma) &&
                        setup_luma();
        }
        
        bool image_t::load_rgba(const string_t& name, const char* buffer, size_t buffer_size)
        {
                return  load_image(name, buffer, buffer_size, color_mode::rgba, m_rgba, m_luma) &&
                        setup_rgba();
        }
        
        bool image_t::load_luma(const string_t& name, const char* buffer, size_t buffer_size)
        {
                return  load_image(name, buffer, buffer_size, color_mode::luma, m_rgba, m_luma) &&
                        setup_luma();
        }

        bool image_t::load_luma(const char* buffer, coord_t rows, coord_t cols)
        {
                const coord_t size = rows * cols;

                m_luma.resize(rows, cols);
                tensor::transform(tensor::make_vector(buffer, size),
                                  m_luma, [] (char luma) { return static_cast<luma_t>(luma); });

                return setup_luma();
        }

        bool image_t::load_rgba(const char* buffer, coord_t rows, coord_t cols, coord_t stride)
        {
                const coord_t size = rows * cols;

                m_rgba.resize(rows, cols);
                tensor::transform(tensor::make_vector(buffer + 0 * stride, size),
                                  tensor::make_vector(buffer + 1 * stride, size),
                                  tensor::make_vector(buffer + 2 * stride, size),
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
                tensor::transform(data, m_luma, [] (rgba_t c) { return color::make_luma(c); });

                return setup_luma();
        }

        bool image_t::load_luma(const luma_matrix_t& data)
        {
                m_luma = data;

                return setup_luma();
        }

        bool image_t::load(const tensor_t& data)
        {
                const scalar_t scale = 255.0;

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
                return save_image(path, m_mode, m_rgba, m_luma);
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

                const scalar_t scale = 1.0 / 255.0;

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
                                        return scale * color::make_red(rgba);
                                });
                                tensor::transform(m_rgba.block(top, left, rows, cols), gmap, [=] (rgba_t rgba)
                                {
                                        return scale * color::make_green(rgba);
                                });
                                tensor::transform(m_rgba.block(top, left, rows, cols), bmap, [=] (rgba_t rgba)
                                {
                                        return scale * color::make_blue(rgba);
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

        bool image_t::copy(coord_t top, coord_t left, const rgba_matrix_t& patch)
        {
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
                                ncv::bilinear(m_luma, luma_scaled, factor);

                                m_luma = luma_scaled;
                        }

                        return setup_luma();

                case color_mode::rgba:
                        {
                                cielab_matrix_t cielab(rows(), cols());
                                tensor::transform(m_rgba, cielab, color::make_cielab);

                                cielab_matrix_t cielab_scaled;
                                ncv::bilinear(cielab, cielab_scaled, factor);

                                m_rgba.resize(cielab_scaled.rows(), cielab_scaled.cols());
                                tensor::transform(cielab_scaled, m_rgba, [] (const cielab_t& lab) { return color::make_rgba(lab); });
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
                        tensor::transform(m_rgba, m_rgba, color::make_opaque);
                        return true;

                default:
                        return false;
                }
        }

        bool image_t::noise(color_channel ch, scalar_t center, scalar_t variance)
        {
                // TODO

                return false;
        }

        bool image_t::smooth(color_channel ch, scalar_t sigma)
        {
                // TODO

                return false;
        }
}
