#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_random_index"

#include <boost/test/unit_test.hpp>
#include "math/random.hpp"

BOOST_AUTO_TEST_CASE(test_random)
{
        const size_t tests = 1024;
        const size_t test_size = 1024;

        for (size_t t = 0; t < tests; ++ t)
        {
                const size_t min = 17 + t;
                const size_t max = min + t * 25 + 4;

                // initialize (uniform) random number generator for the index
                math::random_t<size_t> rgen(min, max);
                math::random_index_t<size_t> rindex(rgen);

                // initialize (uniform) random number generator for the size
                math::random_t<size_t> rsize(0, std::numeric_limits<size_t>::max());

                // check index generator
                for (size_t tt = 0; tt < test_size; ++ tt)
                {
                        const size_t size = rsize();
                        const size_t index = rindex(size);

                        BOOST_CHECK_GE(index, 0);
                        BOOST_CHECK_LT(index, size);
                }
        }
}
