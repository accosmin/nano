#include "ncv_sample.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void sample_t::load_gray(const image_t& image, const rect_t& region)
        {
                const coord_t top = geom::top(region), left = geom::left(region);
                const coord_t rows = geom::rows(region), cols = geom::cols(region);

                m_input.resize(rows * cols);
                for (coord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (coord_t c = 0; c < cols; c ++)
                        {
                                const rgba_t rgba = image.m_rgba(top + r, left + c);
                                m_input(i ++) = color::make_luma(rgba);
                        }
                }
                m_input /= 255.0;

                load_target(image, region);
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_rgba(const image_t& image, const rect_t& region)
        {
                const coord_t top = geom::top(region), left = geom::left(region);
                const coord_t rows = geom::rows(region), cols = geom::cols(region);

                m_input.resize(rows * cols * 3);
                for (coord_t r = 0, i = 0; r < rows; r ++)
                {
                        for (coord_t c = 0; c < cols; c ++)
                        {
                                const rgba_t rgba = image.m_rgba(top + r, left + c);
                                m_input(i ++) = color::make_red(rgba);
                                m_input(i ++) = color::make_green(rgba);
                                m_input(i ++) = color::make_blue(rgba);
                        }
                }
                m_input /= 255.0;

                load_target(image, region);
        }

        //-------------------------------------------------------------------------------------------------

        void sample_t::load_target(const image_t& image, const rect_t& region)
        {
                scalar_t best_overlap = 0.0;
                size_t best_index = 0;

                // find the most overlapping annotation
                for (size_t i = 0; i < image.m_annotations.size(); i ++)
                {
                        const annotation_t& anno = image.m_annotations[i];

                        const scalar_t overlap = geom::overlap(anno.m_region, region);
                        if (overlap > best_overlap)
                        {
                                best_overlap = overlap;
                                best_index = i;
                        }
                }

                static const scalar_t thres_overlap = 0.80;
                if (    best_overlap > thres_overlap &&
                        best_index < image.m_annotations.size())
                {
                        m_target = image.m_annotations[best_index].m_target;
                }
                else
                {
                        m_target.resize(0);
                }
        }

        //-------------------------------------------------------------------------------------------------
}
