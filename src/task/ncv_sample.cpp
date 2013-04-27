#include "ncv_sample.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void annotated_image::load_gray(const char* buffer, size_t rows, size_t cols)
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

        void annotated_image::load_rgba(const char* buffer, size_t rows, size_t cols)
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
}
