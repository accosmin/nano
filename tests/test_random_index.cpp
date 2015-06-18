#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_random_index"

#include <boost/test/unit_test.hpp>
#include "nanocv/math/random.hpp"

BOOST_AUTO_TEST_CASE(test_random)
{
        using namespace ncv;

        const size_t tests = 1024;
        const size_t test_size = 1024;

        for (size_t t = 0; t < tests; t ++)
        {
                const int32_t min = 17 + t;
                const int32_t max = min + t * 25 + 4;

                // initialize (uniform) random number generator for the index
                ncv::random_t<int32_t> rgen(min, max);
                ncv::random_index_t<int32_t> rindex(rgen);

                // initialize (uniform) random number generator for the size
                ncv::random_t<int32_t> rsize(0, std::numeric_limits<int32_t>::max());

                // check index generator
                for (size_t tt = 0; tt < test_size; tt ++)
                {
                        const int32_t size = rsize();
                        const int32_t index = rindex(size);

                        BOOST_CHECK_GE(index, 0);
                        BOOST_CHECK_LT(index, size);
                }
        }
}
