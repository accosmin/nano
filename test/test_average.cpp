#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_average"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "math/epsilon.hpp"
#include "math/average_scalar.hpp"
#include "math/average_vector.hpp"
#include <eigen3/Eigen/Core>

namespace test
{
        template
        <
                typename tscalar,
                typename tsize
        >
        tscalar average(const tsize range)
        {
                return static_cast<tscalar>(range) / static_cast<tscalar>(2);
        }

        template
        <
                typename tscalar,
                typename tsize
        >
        void check_average(const tsize range)
        {
                math::average_scalar_t<tscalar> runavg;
                for (tsize i = 0; i <= range; ++ i)
                {
                        runavg.update(static_cast<tscalar>(i), tscalar(1));
                }

                const auto avg = average<tscalar>(range);

                BOOST_CHECK_LE(math::abs(runavg.value() - avg),
                               math::epsilon1<tscalar>());
        }

        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tsize = typename tvector::Index
        >
        void check_average(const tsize dims, const tsize range)
        {
                math::average_vector_t<tscalar, tvector> runavg(dims);
                for (tsize i = 0; i <= range; ++ i)
                {
                        runavg.update(tvector::Constant(dims, tscalar(i)), tscalar(1));
                }

                const auto avg = average<tscalar>(range);

                BOOST_CHECK_LE((runavg.value() - tvector::Constant(dims, avg)).template lpNorm<Eigen::Infinity>(),
                               math::epsilon1<tscalar>());
        }
}

BOOST_AUTO_TEST_CASE(test_average_scalar)
{
        test::check_average<double>(1);
        test::check_average<double>(5);
        test::check_average<double>(17);
        test::check_average<double>(85);
        test::check_average<double>(187);
        test::check_average<double>(1561);
        test::check_average<double>(14332);
        test::check_average<double>(123434);
}

BOOST_AUTO_TEST_CASE(test_average_vector)
{
        test::check_average<Eigen::VectorXd>(13, 1);
        test::check_average<Eigen::VectorXd>(17, 5);
        test::check_average<Eigen::VectorXd>(11, 17);
        test::check_average<Eigen::VectorXd>(21, 85);
        test::check_average<Eigen::VectorXd>(27, 187);
        test::check_average<Eigen::VectorXd>(15, 1561);
        test::check_average<Eigen::VectorXd>(19, 14332);
        test::check_average<Eigen::VectorXd>(18, 123434);
}

