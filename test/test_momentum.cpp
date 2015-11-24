#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_momentum"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "math/momentum.hpp"
#include "math/epsilon.hpp"
#include <eigen3/Eigen/Core>

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

                        BOOST_CHECK_LE(math::abs(mom00.value() - base00), epsilon);
                        BOOST_CHECK_LE(math::abs(mom01.value() - base01), epsilon);
                        BOOST_CHECK_LE(math::abs(mom10.value() - base10), epsilon);
                        BOOST_CHECK_LE(math::abs(mom11.value() - base11), epsilon);
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

                        BOOST_CHECK_LE((mom00.value() - base00).template lpNorm<Eigen::Infinity>(), epsilon);
                        BOOST_CHECK_LE((mom01.value() - base01).template lpNorm<Eigen::Infinity>(), epsilon);
                        BOOST_CHECK_LE((mom10.value() - base10).template lpNorm<Eigen::Infinity>(), epsilon);
                        BOOST_CHECK_LE((mom11.value() - base11).template lpNorm<Eigen::Infinity>(), epsilon);
                }
        }
}

BOOST_AUTO_TEST_CASE(test_momentum_scalar)
{
        test::check_momentum<double>(0.1, 123);
        test::check_momentum<double>(0.5, 127);
        test::check_momentum<double>(0.9, 253);
}

BOOST_AUTO_TEST_CASE(test_momentum_vector)
{
        test::check_momentum<Eigen::VectorXd>(13, 0.1, 98);
        test::check_momentum<Eigen::VectorXd>(17, 0.5, 75);
        test::check_momentum<Eigen::VectorXd>(11, 0.9, 54);
}

