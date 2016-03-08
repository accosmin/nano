#include "unit_test.hpp"
#include "math/random.hpp"

ZOB_BEGIN_MODULE(test_random)

ZOB_CASE(rng)
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
                        ZOB_CHECK_GREATER_EQUAL(v, min);
                        ZOB_CHECK_LESS_EQUAL(v, max);
                }
        }
}


ZOB_CASE(index)
{
        const int32_t tests = 54;
        const int32_t test_size = 87;

        for (int32_t t = 0; t < tests; ++ t)
        {
                const int32_t min = 17 + t;
                const int32_t max = min + t * 25 + 4;

                auto rgen = math::make_rng(min, max);

                const auto size = rgen();
                auto rindex = math::make_index_rng(size);

                for (size_t tt = 0; tt < test_size; ++ tt)
                {
                        const auto index = rindex();

                        ZOB_CHECK_GREATER_EQUAL(index, 0);
                        ZOB_CHECK_LESS(index, size);
                }
        }
}

ZOB_END_MODULE()
