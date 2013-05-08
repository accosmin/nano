#include "ncv_annotation.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void annotated_image_t::load_gray(const char* buffer, size_t rows, size_t cols)
        {
                m_image.resize(rows, cols);

                for (index_t y = 0, i = 0; y < rows; y ++)
                {
                        for (index_t x = 0; x < cols; x ++, i ++)
                        {
                                m_image(y, x) = color::make_rgba(buffer[i], buffer[i], buffer[i]);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        void annotated_image_t::load_rgba(const char* buffer, size_t rows, size_t cols)
        {
                m_image.resize(rows, cols);

                const size_t size = rows * cols;
                for (size_t y = 0, dr = 0, dg = dr + size, db = dg + size; y < rows; y ++)
                {
                        for (size_t x = 0; x < cols; x ++, dr ++, dg ++, db ++)
                        {
                                m_image(y, x) = color::make_rgba(buffer[dr], buffer[dg], buffer[db]);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        void annotated_image_t::save_gray(const irect_t& region, vector_t& data) const
        {
                const icoord_t l = region.min_corner().x(), r = region.max_corner().x();
                const icoord_t t = region.min_corner().y(), b = region.max_corner().y();
                const icoord_t rows = b - t, cols = r - l;

                data.resize(rows * cols);
                for (icoord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (icoord_t c = 0; c < cols; c ++)
                        {
//                                const cielab_t cielab = color::make_cielab(m_image(t + r, l + c));
//                                data(i ++) = cielab(0);
                                const rgba_t rgba = m_image(t + r, l + c);
                                data(i ++) = color::make_luma(rgba);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        void annotated_image_t::save_rgba(const irect_t& region, vector_t& data) const
        {
                const icoord_t l = region.min_corner().x(), r = region.max_corner().x();
                const icoord_t t = region.min_corner().y(), b = region.max_corner().y();
                const icoord_t rows = b - t, cols = r - l;

                data.resize(rows * cols * 3);
                for (icoord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (icoord_t c = 0; c < cols; c ++)
                        {
//                                const cielab_t cielab = color::make_cielab(m_image(t + r, l + c));
//                                data(i ++) = cielab(0);
//                                data(i ++) = cielab(1);
//                                data(i ++) = cielab(2);
                                const rgba_t rgba = m_image(t + r, l + c);
                                data(i ++) = color::make_red(rgba);
                                data(i ++) = color::make_green(rgba);
                                data(i ++) = color::make_blue(rgba);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------
}
