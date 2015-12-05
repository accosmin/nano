#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_random"

#include <boost/test/unit_test.hpp>
#include "math/random.hpp"

BOOST_AUTO_TEST_CASE(test_random)
{
        const int32_t tests = 231;
        const int32_t test_size = 65;

        for (int32_t t = 0; t < tests; ++ t)
        {
                const int32_t min = 17 + t;
                const int32_t max = min + t * 25 + 4;

                auto rgen = math::make_rng(min, max);

                for (int32_t tt = 0; tt < test_size; ++ tt)
                {
                        const int32_t v = rgen();
                        BOOST_CHECK_GE(v, min);
                        BOOST_CHECK_LE(v, max);
                }
        }
}


BOOST_AUTO_TEST_CASE(test_random_index)
{
        const size_t tests = 54;
        const size_t test_size = 87;

        for (size_t t = 0; t < tests; ++ t)
        {
                const size_t min = 17 + t;
                const size_t max = min + t * 25 + 4;

                auto rgen = math::make_rng(min, max);

                const auto size = rgen();
                auto rindex = math::make_index_rng(size);

                for (size_t tt = 0; tt < test_size; ++ tt)
                {
                        const auto index = rindex();

                        BOOST_CHECK_GE(index, 0);
                        BOOST_CHECK_LT(index, size);
                }
        }
}
