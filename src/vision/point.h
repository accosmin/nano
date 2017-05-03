#pragma once

#include "arch.h"
#include "geom.h"
#include <iosfwd>

namespace nano
{
        ///
        /// \brief 2D point
        ///
        struct NANO_PUBLIC point_t
        {
                ///
                /// \brief constructor
                ///
                explicit point_t(coord_t x = 0, coord_t y = 0) :
                        m_x(x), m_y(y)
                {
                }

                ///
                /// \brief access functions
                ///
                coord_t x() const { return m_x; }
                coord_t y() const { return m_y; }

        private:

                // attributes
                coord_t         m_x;
                coord_t         m_y;
        };

        ///
        /// \brief compare two points
        ///
        NANO_PUBLIC bool operator==(const point_t& point1, const point_t& point2);

        ///
        /// \brief stream a point
        ///
        NANO_PUBLIC std::ostream& operator<<(std::ostream& s, const point_t& point);
}
