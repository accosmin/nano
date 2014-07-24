#ifndef NANOCV_GRID_IMAGE_H
#define NANOCV_GRID_IMAGE_H

#include "color.h"

namespace ncv
{
        // fixme: merge in image_t!

        ///
        /// \brief create an RGBA image composed from fixed-size RGBA patches disposed in a grid
        ///
        class grid_image_t
        {
        public:

                // constructor
                grid_image_t(   size_t patch_rows, size_t patch_cols,
                                size_t group_rows, size_t group_cols,
                                size_t border = 8,
                                rgba_t back_color = color::make_rgba(225, 225, 0));

                // setup a patch at a given grid position
                bool set(size_t grow, size_t gcol, const rgba_matrix_t& patch);

                // access functions
                const rgba_matrix_t& rgba() const { return m_image; }

        private:

                // attributes
                size_t          m_prows;        ///< patch size
                size_t          m_pcols;
                size_t          m_grows;        ///< grid size
                size_t          m_gcols;
                size_t          m_border;       ///< grid border in pixels
                rgba_t          m_bcolor;       ///< background color
                rgba_matrix_t   m_image;
        };
}

#endif //  NANOCV_GRID_IMAGE_H
