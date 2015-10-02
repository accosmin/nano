#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_corr2d"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "core/tensor.h"
#include "math/epsilon.hpp"
#include "tensor/corr2d_cpp.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "tensor/corr2d_egb.hpp"
#include "tensor/corr2d_egr.hpp"

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

        void test_corr2d(int isize, int ksize)
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

                const scalar_t corrcpu_egb = test_cpu(tensor::corr2d_egb_t(), odata, kdata, idata);
                const scalar_t corrcpu_egr = test_cpu(tensor::corr2d_egr_t(), odata, kdata, idata);
                const scalar_t corrcpu_cpp = test_cpu(tensor::corr2d_cpp_t(), odata, kdata, idata);
                const scalar_t corrcpu_mdk = test_cpu(tensor::corr2d_mdk_t(), odata, kdata, idata);
                const scalar_t corrcpu_mdo = test_cpu(tensor::corr2d_mdo_t(), odata, kdata, idata);
                const scalar_t corrcpu_dyn = test_cpu(tensor::corr2d_dyn_t(), odata, kdata, idata);

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                BOOST_CHECK_LE(math::abs(corrcpu_egb - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_egr - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_cpp - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_mdk - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_mdo - corrcpu_egb), epsilon);
                BOOST_CHECK_LE(math::abs(corrcpu_dyn - corrcpu_egb), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_corr2d)
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
                                test::test_corr2d(isize, ksize);
                        }
                }
        }
}

