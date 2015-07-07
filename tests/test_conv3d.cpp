#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_conv2d"

#include <boost/test/unit_test.hpp>
#include "nanocv/tensor.h"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/conv2d.hpp"
#include "nanocv/math/corr2d.hpp"
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

                tensor::conv3d_t<tensor_t> conv3d;
                conv3d.reset(kdata, idims, odims);

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                // output
                const auto op_dyn_output = [&] ()
                {
                        math::conv3d_output(math::conv2d_dyn_t(), idata, kdata, odata);
                        return odata.vector().sum();
                };
                const auto op_lin_output = [&] ()
                {
                        conv3d.output(idata, odata);
                        return odata.vector().sum();
                };

                const auto output_dyn = op_dyn_output();
                const auto output_lin = op_lin_output();

                BOOST_CHECK_LE(math::abs(output_dyn - output_dyn), epsilon);
                BOOST_CHECK_LE(math::abs(output_lin - output_dyn), epsilon);

                // gradient wrt parameters (convolution kernels)
                const auto op_dyn_gparam = [&] ()
                {
                        math::conv3d_gparam(math::conv2d_dyn_t(), idata, kdata, odata);
                        return kdata.vector().sum();
                };
                const auto op_lin_gparam = [&] ()
                {
                        conv3d.gparam(idata, kdata, odata);
                        return kdata.vector().sum();
                };

                const auto gparam_dyn = op_dyn_gparam();
                const auto gparam_lin = op_lin_gparam();

                BOOST_CHECK_LE(math::abs(gparam_dyn - gparam_dyn), epsilon);
                BOOST_CHECK_LE(math::abs(gparam_lin - gparam_dyn), epsilon);

                // gradient wrt inputs
                const auto op_dyn_ginput = [&] ()
                {
                        math::conv3d_ginput(math::corr2d_dyn_t(), idata, kdata, odata);
                        return idata.vector().sum();
                };
                const auto op_lin_ginput = [&] ()
                {
                        conv3d.ginput(idata, odata);
                        return idata.vector().sum();
                };

                const auto ginput_dyn = op_dyn_ginput();
                const auto ginput_lin = op_lin_ginput();

                BOOST_CHECK_LE(math::abs(ginput_dyn - ginput_dyn), epsilon);
                BOOST_CHECK_LE(math::abs(ginput_lin - ginput_dyn), epsilon);
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

