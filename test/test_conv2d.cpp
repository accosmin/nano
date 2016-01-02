#include "unit_test.hpp"
#include "math/abs.hpp"
#include "math/epsilon.hpp"
#include "tensor/matrix.hpp"
#include "tensor/conv2d_cpp.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/conv2d_eig.hpp"

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

                const scalar_t convcpu_eig = test_cpu(tensor::conv2d_eig_t(), idata, kdata, odata);
                const scalar_t convcpu_cpp = test_cpu(tensor::conv2d_cpp_t(), idata, kdata, odata);
                const scalar_t convcpu_dot = test_cpu(tensor::conv2d_dot_t(), idata, kdata, odata);
                const scalar_t convcpu_mad = test_cpu(tensor::conv2d_mad_t(), idata, kdata, odata);
                const scalar_t convcpu_dyn = test_cpu(tensor::conv2d_dyn_t(), idata, kdata, odata);

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                NANOCV_CHECK_CLOSE(convcpu_eig, convcpu_eig, epsilon);
                NANOCV_CHECK_CLOSE(convcpu_cpp, convcpu_eig, epsilon);
                NANOCV_CHECK_CLOSE(convcpu_dot, convcpu_eig, epsilon);
                NANOCV_CHECK_CLOSE(convcpu_mad, convcpu_eig, epsilon);
                NANOCV_CHECK_CLOSE(convcpu_dyn, convcpu_eig, epsilon);
        }
}

NANOCV_BEGIN_MODULE(test_conv2d)

NANOCV_CASE(evaluate)
{
        const int min_isize = 3;
        const int max_isize = 19;
        const int min_ksize = 1;
        const int n_tests = 5;

        for (int isize = min_isize; isize <= max_isize; ++ isize)
        {
                for (int ksize = min_ksize; ksize <= isize - min_ksize; ++ ksize)
                {
                        for (int t = 0; t < n_tests; ++ t)
                        {
                                test::test_conv2d(isize, ksize);
                        }
                }
        }
}

NANOCV_END_MODULE()

