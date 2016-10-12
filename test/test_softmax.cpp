#include "utest.h"
#include "tensor.h"
#include "math/softmax.h"
#include "math/epsilon.h"

using scalar_t = double;//nano::scalar_t;
using vector_t = tensor::vector_t<scalar_t>;

namespace
{
        template <typename tsize>
        vector_t make_vector(const tsize dims)
        {
                return vector_t::Random(dims).array().abs();
        }

        vector_t make_gradient(const vector_t& v)
        {
                int index = 0;
                v.maxCoeff(&index);
                vector_t g = vector_t::Zero(v.size());
                g(index) = 1;
                return g;
        }
}

NANO_BEGIN_MODULE(test_softmax)

NANO_CASE(value)
{
        for (auto test = 0; test < 10; ++ test)
        {
                const auto dims = 10;
                const auto v = make_vector(dims);

                // check that softmax approaches max at the limit
                auto last_diff = std::numeric_limits<scalar_t>::max();
                for (auto beta = scalar_t(1); beta < scalar_t(1025); beta *= scalar_t(1.1))
                {
                        const auto s = nano::softmax_value(v, beta);
                        if (!std::isfinite(s))
                        {
                                break;
                        }

                        const auto m = v.lpNorm<Eigen::Infinity>();
                        const auto diff = std::fabs(m - s) / m;
                        NANO_CHECK_LESS(diff, last_diff + nano::epsilon1<scalar_t>());
                        last_diff = diff;
                }
                NANO_CHECK_LESS(last_diff, nano::epsilon3<scalar_t>());
        }
}

NANO_CASE(vgrad)
{
        for (auto test = 0; test < 10; ++ test)
        {
                const auto dims = 10;
                const auto v = make_vector(dims);
                const auto g = make_gradient(v);

                // check that softmax's gradient approches max's gradient at the limit
                auto last_diff = std::numeric_limits<scalar_t>::max();
                for (auto beta = scalar_t(1); beta < scalar_t(1025); beta *= scalar_t(1.1))
                {
                        const vector_t x = nano::softmax_vgrad(v, beta);
                        if (!std::isfinite(x.sum()))
                        {
                                break;
                        }

                        const auto diff = (g - x).lpNorm<Eigen::Infinity>();
                        NANO_CHECK_LESS(diff, last_diff + nano::epsilon1<scalar_t>());
                        last_diff = diff;
                }
                NANO_CHECK_LESS(last_diff, scalar_t(0.1));//nano::epsilon3<scalar_t>());
        }
}

NANO_END_MODULE()
