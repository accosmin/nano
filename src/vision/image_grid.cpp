#include "image_grid.h"

namespace zob
{
        image_grid_t::image_grid_t(
                coord_t patch_rows, coord_t patch_cols,
                coord_t group_rows, coord_t group_cols,
                coord_t border,
                rgba_t back_color)
                :       m_prows(patch_rows),
                        m_pcols(patch_cols),
                        m_grows(group_rows),
                        m_gcols(group_cols),
                        m_border(border),
                        m_bcolor(back_color)
        {
                const coord_t rows = m_prows * m_grows + m_border * (m_grows + 1);
                const coord_t cols = m_pcols * m_gcols + m_border * (m_gcols + 1);

                m_image.resize(rows, cols, color_mode::rgba);
                m_image.fill(m_bcolor);
        }

        bool image_grid_t::set(coord_t grow, coord_t gcol, const image_t& image)
        {
                return set(grow, gcol, image, rect_t(0, 0, image.cols(), image.rows()));
        }

        bool image_grid_t::set(coord_t grow, coord_t gcol, const image_t& image, const rect_t& region)
        {
                if (    grow < m_grows &&
                        gcol < m_gcols &&
                        region.rows() == m_prows &&
                        region.cols() == m_pcols)
                {
                        const coord_t iy = m_prows * grow + m_border * (grow + 1);
                        const coord_t ix = m_pcols * gcol + m_border * (gcol + 1);

                        return m_image.copy(iy, ix, image, region);
                }
                else
                {
                        return false;
                }
        }
}
