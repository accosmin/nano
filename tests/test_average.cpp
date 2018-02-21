#include "utest.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "tensor/average.h"

namespace test
{
        template <typename tsize>
        tsize sign(const tsize index)
        {
                return (index % 2 == 0) ? tsize(+1) : tsize(-1);
        }

        template <typename tscalar, typename tsize>
        tscalar average1(const tsize range)
        {
                return static_cast<tscalar>(range + 1) / static_cast<tscalar>(2);
        }

        template <typename tscalar, typename tsize>
        tscalar average2(const tsize range)
        {
                return static_cast<tscalar>((range + 1) * sign(range + 1)) / static_cast<tscalar>(2);
        }

        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tsize = typename tvector::Index
        >
        void check_average(const tsize dims, const tsize range)
        {
                nano::average_t<tvector> avg1(dims), avg2(dims);
                for (tsize i = 1; i <= range; ++ i)
                {
                        avg1.update(tvector::Constant(dims, tscalar(i)));
                        avg2.update(tvector::Constant(dims, tscalar(sign(i + 1) * i) * tscalar(i)));
                }

                const auto epsilon = nano::epsilon1<tscalar>();
                const auto base1 = tvector::Constant(dims, average1<tscalar>(range));
                const auto base2 = tvector::Constant(dims, average2<tscalar>(range));

                NANO_CHECK_EIGEN_CLOSE(avg1.value(), base1, epsilon);
                NANO_CHECK_EIGEN_CLOSE(avg2.value(), base2, epsilon);
        }
}

NANO_BEGIN_MODULE(test_average)

NANO_CASE(vector)
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

NANO_END_MODULE()
