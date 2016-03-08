#include "unit_test.hpp"
#include "vision/point.h"

namespace test
{
        void build_point(cortex::coord_t x, cortex::coord_t y)
        {
                const cortex::point_t point(x, y);

                ZOB_CHECK_EQUAL(point.x(), x);
                ZOB_CHECK_EQUAL(point.y(), y);
        }
}

ZOB_BEGIN_MODULE(test_point)

ZOB_CASE(construction)
{
        test::build_point(3, 7);
        test::build_point(7, 3);
        test::build_point(-5, -1);
        test::build_point(-9, +1);
}

ZOB_END_MODULE()
