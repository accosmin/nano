#include "point.h"
#include <iostream>

namespace ncv
{
        std::ostream& operator<<(std::ostream& s, const point_t& point)
        {
                return s << "{POINT: (" << point.x() << ", " << point.y() << "}";
        }
}


