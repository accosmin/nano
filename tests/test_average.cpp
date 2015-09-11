#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_average"

#include <boost/test/unit_test.hpp>
#include "libmath/abs.hpp"
#include "libmath/epsilon.hpp"
#include "liboptim/average_scalar.hpp"

namespace test
{
        using namespace ncv;

        template
        <
                typename tscalar
        >
        void check_average(const size_t range)
        {
                average_scalar_t<tscalar> running_average;

                for (size_t i = 0; i <= range; i ++)
                {
                        running_average.update(static_cast<tscalar>(i), tscalar(1));
                }

                const tscalar real_average = static_cast<tscalar>(range) / static_cast<tscalar>(2);

                BOOST_CHECK_LE(math::abs(running_average.value() - real_average), math::epsilon1<tscalar>());
        }
}

BOOST_AUTO_TEST_CASE(test_average)
{
        test::check_average<double>(1);
        test::check_average<double>(5);
        test::check_average<double>(17);
        test::check_average<double>(85);
        test::check_average<double>(187);
        test::check_average<double>(1561);
        test::check_average<double>(14332);
        test::check_average<double>(123434);
}
