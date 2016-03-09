#pragma once

#include "image.h"

namespace zob
{
        ///
        /// \brief create an RGBA image composed from fixed-size RGBA patches disposed in a grid
        ///
        class ZOB_PUBLIC image_grid_t
        {
        public:

                // constructor
                image_grid_t(   coord_t patch_rows, coord_t patch_cols,
                                coord_t group_rows, coord_t group_cols,
                                coord_t border = 8,
                                rgba_t back_color = color::make_rgba(225, 225, 0));

                // setup a patch at a given grid position
                bool set(coord_t grow, coord_t gcol, const image_t& image);
                bool set(coord_t grow, coord_t gcol, const image_t& image, const rect_t& region);

                // access functions
                const image_t& image() const { return m_image; }

        private:

                // attributes
                coord_t         m_prows;        ///< patch size
                coord_t         m_pcols;
                coord_t         m_grows;        ///< grid size
                coord_t         m_gcols;
                coord_t         m_border;       ///< grid border in pixels
                rgba_t          m_bcolor;       ///< background color
                image_t         m_image;
        };
}

