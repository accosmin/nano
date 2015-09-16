#include "point.h"
#include <iostream>

namespace ncv
{
        bool operator==(const point_t& point1, const point_t& point2)
        {
                return  point1.x() == point2.x() &&
                        point1.y() == point2.y();
        }

        std::ostream& operator<<(std::ostream& s, const point_t& point)
        {
                return s << "{POINT: (" << point.x() << ", " << point.y() << "}";
        }
}


