#include "unit_test.hpp"
#include "math/momentum.hpp"
#include "math/epsilon.hpp"

namespace test
{        
        template
        <
                typename tscalar,
                typename tsize
        >
        void check_momentum(const tscalar momentum, const tsize range)
        {
                math::momentum_scalar_t<tscalar> mom00(momentum, tscalar(0));
                math::momentum_scalar_t<tscalar> mom01(momentum, tscalar(1));
                math::momentum_scalar_t<tscalar> mom10(momentum, tscalar(0));
                math::momentum_scalar_t<tscalar> mom11(momentum, tscalar(1));

                const auto epsilon = math::epsilon1<tscalar>();
                auto powm = momentum;
                for (tsize i = 1; i <= range; ++ i, powm *= momentum)
                {
                        mom00.update(tscalar(0));
                        mom01.update(tscalar(0));
                        mom10.update(tscalar(1));
                        mom11.update(tscalar(1));

                        const auto base00 = tscalar(0);
                        const auto base01 = powm;
                        const auto base10 = tscalar(1) - powm;
                        const auto base11 = tscalar(1);

                        NANOCV_CHECK_CLOSE(mom00.value(), base00, epsilon);
                        NANOCV_CHECK_CLOSE(mom01.value(), base01, epsilon);
                        NANOCV_CHECK_CLOSE(mom10.value(), base10, epsilon);
                        NANOCV_CHECK_CLOSE(mom11.value(), base11, epsilon);
                }
        }

        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tsize = typename tvector::Index
        >
        void check_momentum(const tsize dims, const tscalar momentum, const tsize range)
        {
                math::momentum_vector_t<tvector> mom00(momentum, tvector::Constant(dims, tscalar(0)));
                math::momentum_vector_t<tvector> mom01(momentum, tvector::Constant(dims, tscalar(1)));
                math::momentum_vector_t<tvector> mom10(momentum, tvector::Constant(dims, tscalar(0)));
                math::momentum_vector_t<tvector> mom11(momentum, tvector::Constant(dims, tscalar(1)));

                const auto epsilon = math::epsilon1<tscalar>();
                auto powm = momentum;
                for (tsize i = 1; i <= range; ++ i, powm *= momentum)
                {
                        mom00.update(tvector::Constant(dims, tscalar(0)));
                        mom01.update(tvector::Constant(dims, tscalar(0)));
                        mom10.update(tvector::Constant(dims, tscalar(1)));
                        mom11.update(tvector::Constant(dims, tscalar(1)));

                        const auto base00 = tvector::Constant(dims, tscalar(0));
                        const auto base01 = tvector::Constant(dims, powm);
                        const auto base10 = tvector::Constant(dims, tscalar(1) - powm);
                        const auto base11 = tvector::Constant(dims, tscalar(1));

                        NANOCV_CHECK_EIGEN_CLOSE(mom00.value(), base00, epsilon);
                        NANOCV_CHECK_EIGEN_CLOSE(mom01.value(), base01, epsilon);
                        NANOCV_CHECK_EIGEN_CLOSE(mom10.value(), base10, epsilon);
                        NANOCV_CHECK_EIGEN_CLOSE(mom11.value(), base11, epsilon);
                }
        }
}

NANOCV_BEGIN_MODULE(test_momentum)

NANOCV_CASE(scalar)
{
        test::check_momentum<double>(0.1, 123);
        test::check_momentum<double>(0.5, 127);
        test::check_momentum<double>(0.9, 253);
}

NANOCV_CASE(vector)
{
        test::check_momentum<Eigen::VectorXd>(13, 0.1, 98);
        test::check_momentum<Eigen::VectorXd>(17, 0.5, 75);
        test::check_momentum<Eigen::VectorXd>(11, 0.9, 54);
}

NANOCV_END_MODULE()
