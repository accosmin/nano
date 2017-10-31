#pragma once

#include "image.h"

namespace nano
{
        ///
        /// \brief create an RGBA image composed from fixed-size RGBA patches disposed in a grid.
        ///
        class NANO_PUBLIC image_grid_t
        {
        public:

                // constructor
                image_grid_t(   const coord_t patch_rows, const coord_t patch_cols,
                                const coord_t group_rows, const coord_t group_cols,
                                const coord_t border = 8,
                                const rgba_t& back_color = {225, 225, 0, 255});

                // setup a patch at a given grid position
                bool set(coord_t grow, coord_t gcol, const image_t& image);

                // access functions
                const image_t& image() const { return m_image; }

        private:

                // attributes
                coord_t         m_prows;        ///< patch size
                coord_t         m_pcols;
                coord_t         m_grows;        ///< grid size
                coord_t         m_gcols;
                coord_t         m_border;       ///< grid border in pixels
                image_t         m_image;
        };
}

