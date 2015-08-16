#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_conv2d"

#include <boost/test/unit_test.hpp>
#include "nanocv/tensor.h"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/epsilon.hpp"
#include "nanocv/math/conv2d_cpp.hpp"
#include "nanocv/math/conv2d_dyn.hpp"
#include "nanocv/math/conv2d_eig.hpp"

namespace test
{
        using namespace ncv;

        template
        <
                typename top,
                typename tmatrix
        >
        decltype(auto) test_cpu(top op, const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
        {
                odata.setZero();

                op(idata, kdata, odata);

                return odata.sum();
        }

        void test_conv2d(int isize, int ksize)
        {
                const int osize = isize - ksize + 1;

                matrix_t idata(isize, isize);
                matrix_t kdata(ksize, ksize);
                matrix_t odata(osize, osize);

                idata.setRandom();
                kdata.setRandom();
                odata.setRandom();

                idata /= isize;
                kdata /= ksize;
                odata /= osize;

                const scalar_t convcpu_eig = test_cpu(ncv::math::conv2d_eig_t(), idata, kdata, odata);
                const scalar_t convcpu_cpp = test_cpu(ncv::math::conv2d_cpp_t(), idata, kdata, odata);
                const scalar_t convcpu_dot = test_cpu(ncv::math::conv2d_dot_t(), idata, kdata, odata);
                const scalar_t convcpu_mad = test_cpu(ncv::math::conv2d_mad_t(), idata, kdata, odata);
                const scalar_t convcpu_dyn = test_cpu(ncv::math::conv2d_dyn_t(), idata, kdata, odata);

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                BOOST_CHECK_LE(math::abs(convcpu_eig - convcpu_eig), epsilon);
                BOOST_CHECK_LE(math::abs(convcpu_cpp - convcpu_eig), epsilon);
                BOOST_CHECK_LE(math::abs(convcpu_dot - convcpu_eig), epsilon);
                BOOST_CHECK_LE(math::abs(convcpu_mad - convcpu_eig), epsilon);
                BOOST_CHECK_LE(math::abs(convcpu_dyn - convcpu_eig), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_conv2d)
{
        using namespace ncv;

        const int min_isize = 12;
        const int max_isize = 48;
        const int min_ksize = 1;
        const int n_tests = 16;

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                for (int ksize = min_ksize; ksize <= isize - min_ksize; ksize ++)
                {
                        for (int t = 0; t < n_tests; t ++)
                        {
                                test::test_conv2d(isize, ksize);
                        }
                }
        }
}

