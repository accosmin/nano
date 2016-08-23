#include "utest.hpp"
#include "math/softmax.hpp"
#include "math/epsilon.hpp"

namespace test
{
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tsize = typename tvector::Index
        >
        void check_softmax(const tsize dims, const tscalar beta)
        {
                const tvector v = tvector::Random(dims);
                const tscalar s = nano::softmax_value(v, beta);
                const tscalar diff = std::fabs(v.template lpNorm<Eigen::Infinity>() - s);
                std::cout << "beta = " << beta << ", diff = " << diff << std::endl;
        }
}

NANO_BEGIN_MODULE(test_softmax)

NANO_CASE(value)
{
        for (auto test = 0; test < 10; ++ test)
        {
                const auto dims = 10;
                const auto v = Eigen::VectorXd::Random(dims);

                for (auto beta = 1.0; beta < 1000.0; beta *= 2.0)
                {
                        const auto s = nano::softmax_value(v, beta);
                        NANO_CHECK(std::isfinite(s));
                        const auto diff = std::fabs(v.lpNorm<Eigen::Infinity>() - s);
                        std::cout << "beta = " << beta << ", diff = " << diff << std::endl;
                }
        }
}

// todo: check why the error does not always decrease with 1/beta
// todo: check also the gradient

NANO_END_MODULE()
