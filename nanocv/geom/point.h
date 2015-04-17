#pragma once

#include "geom.h"
#include "../arch.h"
#include <iosfwd>

namespace ncv
{
        ///
        /// \brief 2D point
        ///
        class NANOCV_PUBLIC point_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit point_t(coord_t x = 0, coord_t y = 0)
                        :       m_x(x), m_y(y)
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
        inline bool operator==(const point_t& point1, const point_t& point2)
        {
                return  point1.x() == point2.x() &&
                        point1.y() == point2.y();
        }

        ///
        /// \brief stream a point
        ///
        NANOCV_PUBLIC std::ostream& operator<<(std::ostream& s, const point_t& point);
}


