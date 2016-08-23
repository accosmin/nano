#include "utest.hpp"
#include "tensor.h"
#include "math/softmax.hpp"
#include "math/epsilon.hpp"

NANO_BEGIN_MODULE(test_softmax)

NANO_CASE(value)
{
        using namespace nano;

        for (auto test = 0; test < 10; ++ test)
        {
                const auto dims = 10;
                const vector_t v = vector_t::Random(dims).array().abs();

                auto last_diff = std::numeric_limits<scalar_t>::max();
                for (auto beta = scalar_t(1); beta < scalar_t(65); beta *= 2)
                {
                        const auto s = nano::softmax_value(v, beta);
                        NANO_CHECK(std::isfinite(s));

                        const auto m = v.lpNorm<Eigen::Infinity>();
                        const auto diff = std::fabs(m - s) / m;
                        NANO_CHECK_LESS(diff, last_diff);
                        last_diff = diff;
                }
                NANO_CHECK_LESS(last_diff, nano::epsilon3<scalar_t>());
        }
}

NANO_CASE(vgrad)
{
        // todo: check gradient
        using namespace nano;

        for (auto test = 0; test < 10; ++ test)
        {
                const auto dims = 10;
                const vector_t v = vector_t::Random(dims).array().abs();

                auto last_diff = std::numeric_limits<scalar_t>::max();
                for (auto beta = scalar_t(1); beta < scalar_t(65); beta *= 2)
                {
                        const auto s = nano::softmax_value(v, beta);
                        NANO_CHECK(std::isfinite(s));

                        const auto m = v.lpNorm<Eigen::Infinity>();
                        const auto diff = std::fabs(m - s) / m;
                        NANO_CHECK_LESS(diff, last_diff);
                        last_diff = diff;
                }
                NANO_CHECK_LESS(last_diff, nano::epsilon3<scalar_t>());
        }
}

NANO_END_MODULE()
