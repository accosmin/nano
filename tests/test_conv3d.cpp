#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_conv3d"

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

                random_t<scalar_t> rng(-1.0 / isize, 1.0 / isize);

                tensor_t idata(idims, isize, isize);
                tensor_t kdata(kdims, ksize, ksize);
                tensor_t odata(odims, osize, osize);

                tensor::set_random(idata, rng);
                tensor::set_random(kdata, rng);
                tensor::set_random(odata, rng);

                tensor_t idata_dyn = idata, idata_lin = idata;
                tensor_t kdata_dyn = kdata, kdata_lin = kdata;
                tensor_t odata_dyn = odata, odata_lin = odata;

                tensor::conv3d_t<tensor_t> conv3d;
                BOOST_CHECK(conv3d.reset(kdata, idims, odims));

                const scalar_t epsilon = math::epsilon1<scalar_t>();

                // 2D convolution-based
                math::conv3d_output(math::conv2d_dyn_t(), idata, kdata, odata_dyn);
                math::conv3d_gparam(math::conv2d_dyn_t(), idata, kdata_dyn, odata);
                math::conv3d_ginput(math::corr2d_dyn_t(), idata_dyn, kdata, odata);

                // linearized tensors-based
                BOOST_CHECK(conv3d.output(idata, odata_lin));
                BOOST_CHECK(conv3d.gparam(idata, kdata_lin, odata));
                BOOST_CHECK(conv3d.ginput(idata_lin, odata));

                // check results
                BOOST_CHECK_LE((odata_dyn.vector() - odata_lin.vector()).lpNorm<Eigen::Infinity>(), epsilon);
                BOOST_CHECK_LE((kdata_dyn.vector() - kdata_lin.vector()).lpNorm<Eigen::Infinity>(), epsilon);
                BOOST_CHECK_LE((idata_dyn.vector() - idata_lin.vector()).lpNorm<Eigen::Infinity>(), epsilon);
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

