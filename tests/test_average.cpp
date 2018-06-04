#include "utest.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "tensor/average.h"

template <typename tsize>
tsize sign(const tsize index)
{
        return (index % 2 == 0) ? tsize(+1) : tsize(-1);
}

static Eigen::VectorXd average1(const int dims, const int range)
{
        return Eigen::VectorXd::Constant(dims, static_cast<double>(range + 1) / 2.0);
}

static Eigen::VectorXd average2(const int dims, const int range)
{
        return Eigen::VectorXd::Constant(dims, static_cast<double>((range + 1) * sign(range + 1)) / 2.0);
}

static auto nano_average1(const int dims, const int range)
{
        nano::average_t<Eigen::VectorXd> avg1(dims);
        for (auto i = 1; i <= range; ++ i)
        {
                avg1.update(Eigen::VectorXd::Constant(dims, double(i)));
        }

        return avg1;
}

static auto nano_average2(const int dims, const int range)
{
        nano::average_t<Eigen::VectorXd> avg2(dims);
        for (auto i = 1; i <= range; ++ i)
        {
                avg2.update(Eigen::VectorXd::Constant(dims, double(sign(i + 1) * i) * double(i)));
        }

        return avg2;
}

NANO_BEGIN_MODULE(test_average)

NANO_CASE(vectorXd_13_1)
{
        const auto dims = 13, range = 1;
        NANO_CHECK_EIGEN_CLOSE(nano_average1(dims, range).value(), average1(dims, range), nano::epsilon1<double>());
        NANO_CHECK_EIGEN_CLOSE(nano_average2(dims, range).value(), average2(dims, range), nano::epsilon1<double>());
}

NANO_CASE(vectorXd_17_5)
{
        const auto dims = 17, range = 5;
        NANO_CHECK_EIGEN_CLOSE(nano_average1(dims, range).value(), average1(dims, range), nano::epsilon1<double>());
        NANO_CHECK_EIGEN_CLOSE(nano_average2(dims, range).value(), average2(dims, range), nano::epsilon1<double>());
}

NANO_CASE(vectorXd_11_17)
{
        const auto dims = 11, range = 17;
        NANO_CHECK_EIGEN_CLOSE(nano_average1(dims, range).value(), average1(dims, range), nano::epsilon1<double>());
        NANO_CHECK_EIGEN_CLOSE(nano_average2(dims, range).value(), average2(dims, range), nano::epsilon1<double>());
}

NANO_CASE(vectorXd_21_85)
{
        const auto dims = 21, range = 85;
        NANO_CHECK_EIGEN_CLOSE(nano_average1(dims, range).value(), average1(dims, range), nano::epsilon1<double>());
        NANO_CHECK_EIGEN_CLOSE(nano_average2(dims, range).value(), average2(dims, range), nano::epsilon1<double>());
}

NANO_CASE(vectorXd_27_187)
{
        const auto dims = 27, range = 187;
        NANO_CHECK_EIGEN_CLOSE(nano_average1(dims, range).value(), average1(dims, range), nano::epsilon1<double>());
        NANO_CHECK_EIGEN_CLOSE(nano_average2(dims, range).value(), average2(dims, range), nano::epsilon1<double>());
}

NANO_CASE(vectorXd_15_1561)
{
        const auto dims = 15, range = 1561;
        NANO_CHECK_EIGEN_CLOSE(nano_average1(dims, range).value(), average1(dims, range), nano::epsilon1<double>());
        NANO_CHECK_EIGEN_CLOSE(nano_average2(dims, range).value(), average2(dims, range), nano::epsilon1<double>());
}

NANO_END_MODULE()
