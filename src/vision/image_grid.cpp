#include "image_grid.h"

using namespace nano;

image_grid_t::image_grid_t(
        const coord_t patch_rows, const coord_t patch_cols,
        const coord_t group_rows, const coord_t group_cols,
        const coord_t border,
        const rgba_t back_color) :
        m_prows(patch_rows),
        m_pcols(patch_cols),
        m_grows(group_rows),
        m_gcols(group_cols),
        m_border(border)
{
        const coord_t rows = m_prows * m_grows + m_border * (m_grows + 1);
        const coord_t cols = m_pcols * m_gcols + m_border * (m_gcols + 1);

        m_image.resize(rows, cols, color_mode::rgba);
        m_image.fill(back_color);
}

bool image_grid_t::set(const coord_t grow, const coord_t gcol, const image_t& image)
{
        if (    grow < m_grows &&
                gcol < m_gcols &&
                image.dims() == m_image.dims() &&
                image.rows() == m_prows &&
                image.cols() == m_pcols)
        {
                const coord_t iy = m_prows * grow + m_border * (grow + 1);
                const coord_t ix = m_pcols * gcol + m_border * (gcol + 1);

                for (auto i = 0; i < image.dims(); ++ i)
                {
                        m_image.plane(i, iy, ix, m_prows, m_pcols) = image.plane(i);
                }

                return true;
        }
        else
        {
                return false;
        }
}
