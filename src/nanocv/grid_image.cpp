#include "grid_image.h"

namespace ncv
{
        grid_image_t::grid_image_t(
                size_t patch_rows, size_t patch_cols,
                size_t group_rows, size_t group_cols,
                size_t border,
                rgba_t back_color)
                :       m_prows(patch_rows),
                        m_pcols(patch_cols),
                        m_grows(group_rows),
                        m_gcols(group_cols),
                        m_border(border),
                        m_bcolor(back_color)
        {
                const size_t rows = m_prows * m_grows + m_border * (m_grows + 1);
                const size_t cols = m_pcols * m_gcols + m_border * (m_gcols + 1);

                m_image.resize(rows, cols);
                m_image.setConstant(m_bcolor);
        }

        bool grid_image_t::set(size_t grow, size_t gcol, const rgba_matrix_t& patch)
        {
                if (    grow < m_grows &&
                        gcol < m_gcols &&
                        static_cast<size_t>(patch.rows()) == m_prows &&
                        static_cast<size_t>(patch.cols()) == m_pcols)
                {
                        const size_t iy = m_prows * grow + m_border * (grow + 1);
                        const size_t ix = m_pcols * gcol + m_border * (gcol + 1);
                        const size_t ih = m_prows;
                        const size_t iw = m_pcols;

                        m_image.block(iy, ix, ih, iw) = patch;
                        return true;
                }
                else
                {
                        return false;
                }
        }
}
