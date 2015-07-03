#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_conv2d"

#include <boost/test/unit_test.hpp>
#include "nanocv/tensor.h"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/conv3d.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/math/epsilon.hpp"
#include "nanocv/tensor/conv3d.hpp"

namespace test
{
        using namespace ncv;

        void test_conv3d(int isize, int idims, int ksize, int odims)
        {
                const int osize = isize - ksize + 1;
                const int kdims = odims * idims;

                tensor_t idata(idims, isize, isize);
                tensor_t kdata(kdims, ksize, ksize);
                tensor_t odata(odims, osize, osize);

                random_t<scalar_t> rng(-1.0, 1.0);

                idata.setRandom(rng);
                kdata.setRandom(rng);
                odata.setRandom(rng);

                idata.vector() /= isize;
                kdata.vector() /= ksize;
                odata.vector() /= osize;                

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                // output
                const auto op_dyn_output = [&] ()
                {
                        math::conv3d_output(idata, kdata, odata);
                        return odata.vector().sum();
                };
                const auto op_toe_output = [&] ()
                {
                        tensor::conv3d_output(idata, kdata, odata);
                        return odata.vector().sum();
                };

                const auto output_dyn = op_dyn_output();
                const auto output_toe = op_toe_output();

                BOOST_CHECK_LE(math::abs(output_dyn - output_dyn), epsilon);
                BOOST_CHECK_LE(math::abs(output_toe - output_dyn), epsilon);

                // gradient wrt parameters (convolution kernels)
                const auto op_dyn_gparam = [&] ()
                {
                        math::conv3d_gparam(idata, kdata, odata);
                        return kdata.vector().sum();
                };
                const auto op_toe_gparam = [&] ()
                {
                        tensor::conv3d_gparam(idata, kdata, odata);
                        return kdata.vector().sum();
                };

                const auto gparam_dyn = op_dyn_gparam();
                const auto gparam_toe = op_toe_gparam();

                BOOST_CHECK_LE(math::abs(gparam_dyn - gparam_dyn), epsilon);
                BOOST_CHECK_LE(math::abs(gparam_toe - gparam_dyn), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_conv3d)
{
        using namespace ncv;

        const int min_isize = 4;
        const int max_isize = 16;

        const int min_ksize = 1;
        const int max_ksize = 9;

        const int idims = 16;
        const int odims = 32;

        const int n_tests = 16;

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                for (int ksize = min_ksize; ksize <= std::min(max_ksize, isize); ksize ++)
                {
                        for (int t = 0; t < n_tests; t ++)
                        {
                                test::test_conv3d(isize, idims, ksize, odims);
                        }
                }
        }
}

