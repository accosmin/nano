#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_range"

#include <boost/test/unit_test.hpp>
#include "libnanocv/range.hpp"
#include "libnanocv/random.hpp"

namespace test
{
        using namespace ncv;

        void check_range(double min, double max, size_t tests)
        {
                range_t<double> range(min, max);

                BOOST_CHECK_LT(min, max);
                BOOST_CHECK_EQUAL(range.min(), min);
                BOOST_CHECK_EQUAL(range.max(), max);

                // check random values
                for (size_t i = 0; i < tests; i ++)
                {
                        random_t<double> rgen(min - 0.1, max + 0.56);

                        const auto val = rgen();
                        if (val < min)
                        {
                                BOOST_CHECK_EQUAL(range.clamp(val), min);
                        }
                        else if (val > max)
                        {
                                BOOST_CHECK_EQUAL(range.clamp(val), max);
                        }
                        else
                        {
                                BOOST_CHECK_EQUAL(range.clamp(val), val);
                        }
                }
        }
}

BOOST_AUTO_TEST_CASE(test_range)
{
        test::check_range(-0.03, 0.005, 32);
        test::check_range(1.03, 13.005, 37);
        test::check_range(-0.54, 0.105, 13);
        test::check_range(-7.03, 10.005, 11);
}
