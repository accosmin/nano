#include "utest.h"
#include "math/random.h"

NANO_BEGIN_MODULE(test_random)

NANO_CASE(rng)
{
        const int32_t tests = 231;
        const int32_t test_size = 65;

        for (int32_t t = 0; t < tests; ++ t)
        {
                const int32_t min = 17 + t;
                const int32_t max = min + t * 25 + 4;

                auto rgen = nano::make_rng(min, max);

                for (int32_t tt = 0; tt < test_size; ++ tt)
                {
                        const int32_t v = rgen();
                        NANO_CHECK_GREATER_EQUAL(v, min);
                        NANO_CHECK_LESS_EQUAL(v, max);
                }
        }
}


NANO_CASE(index)
{
        const int32_t tests = 54;
        const int32_t test_size = 87;

        for (int32_t t = 0; t < tests; ++ t)
        {
                const int32_t min = 17 + t;
                const int32_t max = min + t * 25 + 4;

                auto rgen = nano::make_rng(min, max);

                const auto size = rgen();
                auto rindex = nano::make_index_rng(size);

                for (size_t tt = 0; tt < test_size; ++ tt)
                {
                        const auto index = rindex();

                        NANO_CHECK_GREATER_EQUAL(index, 0);
                        NANO_CHECK_LESS(index, size);
                }
        }
}

NANO_END_MODULE()
