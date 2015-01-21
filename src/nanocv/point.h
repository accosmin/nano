#pragma once

#include "geom.h"

namespace ncv
{
        ///
        /// \brief image coordinate
        ///
        typedef int16_t                 coord_t;
        typedef int32_t                 area_t;

        ///
        /// \brief 2D point
        ///
        class point_t
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
        template
        <
                typename tstreamer
        >
        tstreamer& operator<<(tstreamer& s, const point_t& point)
        {
                return s << "{POINT: (" << point.x() << ", " << point.y() << "}";
        }
}


