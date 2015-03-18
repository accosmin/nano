#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_average"

#include <boost/test/unit_test.hpp>
#include "libnanocv/optimize/average.hpp"
#include "libnanocv/math/epsilon.hpp"
#include "libnanocv/math/abs.hpp"

namespace test
{
        using namespace ncv;

        void check_average(size_t range)
        {
                typedef double test_scalar_t;

                optimize::average_scalar<test_scalar_t> running_average;

                for (size_t i = 0; i <= range; i ++)
                {
                        running_average.update(static_cast<test_scalar_t>(i), test_scalar_t(1));
                }

                const test_scalar_t real_average = static_cast<test_scalar_t>(range) / static_cast<test_scalar_t>(2);

                BOOST_CHECK_LE(math::abs(running_average.value() - real_average), math::epsilon1<test_scalar_t>());
        }
}

BOOST_AUTO_TEST_CASE(test_average)
{
        test::check_average(1);
        test::check_average(5);
        test::check_average(17);
        test::check_average(85);
        test::check_average(187);
        test::check_average(1561);
        test::check_average(14332);
        test::check_average(123434);
}
