#include "ncv_sample.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void sample_t::load_gray(const image_t& image, const rect_t& region)
        {
                const coord_t top = get_top(region), left = get_left(region);
                const coord_t rows = get_rows(region), cols = get_cols(region);

                m_data.resize(rows * cols);
                for (coord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (coord_t c = 0; c < cols; c ++)
                        {
                                const rgba_t rgba = image.m_rgba(top + r, left + c);
                                m_data(i ++) = color::make_luma(rgba);
                        }
                }
                m_data /= 255.0;

                load_target(image, region);
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_rgba(const image_t& image, const rect_t& region)
        {
                const coord_t top = get_top(region), left = get_left(region);
                const coord_t rows = get_rows(region), cols = get_cols(region);

                m_data.resize(rows * cols * 3);
                for (coord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (coord_t c = 0; c < cols; c ++)
                        {
                                const rgba_t rgba = image.m_rgba(top + r, left + c);
                                m_data(i ++) = color::make_red(rgba);
                                m_data(i ++) = color::make_green(rgba);
                                m_data(i ++) = color::make_blue(rgba);
                        }
                }
                m_data /= 255.0;

                load_target(image, region);
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_target(const image_t& image, const rect_t& /*region*/)
        {
                // FIXME: assuming image classification (max one annotation for the whole image)!
                if (image.m_annotations.empty())
                {
                        m_target.resize(0);
                }
                else
                {
                        m_target = image.m_annotations[0].m_target;
                }
        }

        //-------------------------------------------------------------------------------------------------
}
