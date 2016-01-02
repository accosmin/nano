#include "unit_test.hpp"
#include "cortex/vision/point.h"

namespace test
{
        void build_point(cortex::coord_t x, cortex::coord_t y)
        {
                const cortex::point_t point(x, y);

                NANOCV_CHECK_EQUAL(point.x(), x);
                NANOCV_CHECK_EQUAL(point.y(), y);
        }
}

NANOCV_BEGIN_MODULE(test_point)

NANOCV_CASE(construction)
{
        test::build_point(3, 7);
        test::build_point(7, 3);
        test::build_point(-5, -1);
        test::build_point(-9, +1);
}

NANOCV_END_MODULE()
