#include "utest.h"
#include "math/epsilon.h"
#include "tensor/momentum.h"

namespace test
{
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tsize = typename tvector::Index
        >
        void check_momentum(const tsize dims, const tscalar momentum, const tsize range)
        {
                nano::momentum_t<tvector> mom00(momentum, dims);
                nano::momentum_t<tvector> mom01(momentum, dims);
                nano::momentum_t<tvector> mom10(1 - momentum, dims);
                nano::momentum_t<tvector> mom11(1 - momentum, dims);

                const auto epsilon = nano::epsilon1<tscalar>();
                for (tsize i = 1; i <= range; ++ i)
                {
                        const auto base00 = tvector::Constant(dims, momentum);
                        const auto base01 = tvector::Constant(dims, 1 - momentum);
                        const auto base10 = tvector::Constant(dims, momentum);
                        const auto base11 = tvector::Constant(dims, 1 - momentum);

                        mom00.update(base00);
                        mom01.update(base01);
                        mom10.update(base10);
                        mom11.update(base11);

                        NANO_CHECK_EIGEN_CLOSE(mom00.value(), base00, epsilon);
                        NANO_CHECK_EIGEN_CLOSE(mom01.value(), base01, epsilon);
                        NANO_CHECK_EIGEN_CLOSE(mom10.value(), base10, epsilon);
                        NANO_CHECK_EIGEN_CLOSE(mom11.value(), base11, epsilon);
                }
        }
}

NANO_BEGIN_MODULE(test_momentum)

NANO_CASE(vector)
{
        test::check_momentum<Eigen::VectorXf>(13, 0.1f, 98);
        test::check_momentum<Eigen::VectorXd>(17, 0.5, 75);
        test::check_momentum<Eigen::VectorXd>(11, 0.9, 54);
}

NANO_END_MODULE()
