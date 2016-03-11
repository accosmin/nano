#include "unit_test.hpp"
#include "math/abs.hpp"
#include "math/epsilon.hpp"
#include "tensor/matrix.hpp"
#include "tensor/corr2d_cpp.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "tensor/corr2d_egb.hpp"
#include "tensor/corr2d_egr.hpp"

namespace test
{
        typedef double scalar_t;
        typedef tensor::matrix_t<scalar_t> matrix_t;

        template
        <
                typename top,
                typename tmatrix
        >
        auto test_cpu(top op, const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
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
                const scalar_t corrcpu_mdk_dyn = test_cpu(tensor::corr2d_mdk_dyn_t(), odata, kdata, idata);
                const scalar_t corrcpu_mdo_dyn = test_cpu(tensor::corr2d_mdo_dyn_t(), odata, kdata, idata);

                const scalar_t epsilon = nano::epsilon1<scalar_t>();

                NANO_CHECK_CLOSE(corrcpu_egb, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_egr, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_cpp, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_mdk, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_mdo, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_dyn, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_mdk_dyn, corrcpu_egb, epsilon);
                NANO_CHECK_CLOSE(corrcpu_mdo_dyn, corrcpu_egb, epsilon);
        }
}

NANO_BEGIN_MODULE(test_corr2d)

NANO_CASE(evaluate)
{
        const int min_isize = 3;
        const int max_isize = 23;
        const int min_ksize = 1;
        
        for (int isize = min_isize; isize <= max_isize; ++ isize)
        {
                for (int ksize = min_ksize; ksize <= isize - min_ksize; ++ ksize)
                {
                        test::test_corr2d(isize, ksize);
                }
        }
}

NANO_END_MODULE()

