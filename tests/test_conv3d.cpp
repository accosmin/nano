#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_conv3d"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "core/tensor.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "tensor/random.hpp"
#include "tensor/conv3d.hpp"
#include "tensor/conv3d_lin.hpp"
#include "tensor/conv2d_cpp.hpp"
#include "tensor/conv2d_dyn.hpp"
#include "tensor/conv2d_eig.hpp"
#include "tensor/corr2d_cpp.hpp"
#include "tensor/corr2d_dyn.hpp"
#include "tensor/corr2d_egb.hpp"
#include "tensor/corr2d_egr.hpp"

namespace test
{
        using namespace ncv;

        template
        <
                typename ttensor,
                typename tscalar = typename ttensor::Scalar
        >
        void test_conv3d(int isize, int idims, int ksize, int odims)
        {
                const int osize = isize - ksize + 1;
                const int kdims = odims * idims;

                math::random_t<tscalar> rng(-1.0 / isize, 1.0 / isize);

                ttensor idata(idims, isize, isize);
                ttensor kdata(kdims, ksize, ksize);
                ttensor odata(odims, osize, osize);

                tensor::set_random(idata, rng);
                tensor::set_random(kdata, rng);
                tensor::set_random(odata, rng);

                ttensor idata_dyn = idata, idata_lin = idata;
                ttensor kdata_dyn = kdata, kdata_lin = kdata;
                ttensor odata_dyn = odata, odata_lin = odata;

                tensor::conv3d_lin_t<ttensor> conv3d;
                BOOST_CHECK(conv3d.reset(kdata, idims, odims));

                const tscalar epsilon = math::epsilon1<tscalar>();

                // 2D convolution-based
                tensor::conv3d_output(tensor::conv2d_dyn_t(), idata, kdata, odata_dyn);
                tensor::conv3d_gparam(tensor::conv2d_dyn_t(), idata, kdata_dyn, odata);
                tensor::conv3d_ginput(tensor::corr2d_dyn_t(), idata_dyn, kdata, odata);

                // linearized tensors-based
                BOOST_CHECK(conv3d.output(idata, odata_lin));
                BOOST_CHECK(conv3d.gparam(idata, kdata_lin, odata));
                BOOST_CHECK(conv3d.ginput(idata_lin, odata));

                // check results
                BOOST_CHECK_LE((odata_dyn.vector() - odata_lin.vector()).template lpNorm<Eigen::Infinity>(), epsilon);
                BOOST_CHECK_LE((kdata_dyn.vector() - kdata_lin.vector()).template lpNorm<Eigen::Infinity>(), epsilon);
                BOOST_CHECK_LE((idata_dyn.vector() - idata_lin.vector()).template lpNorm<Eigen::Infinity>(), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_conv3d)
{
        using namespace ncv;

        const int min_isize = 4;
        const int max_isize = 32;

        const int min_ksize = 1;
        const int max_ksize = 15;

        const int idims = 4;
        const int odims = 8;

        const int n_tests = 8;

        for (int isize = min_isize; isize <= max_isize; isize += 4)
        {
                for (int ksize = min_ksize; ksize <= std::min(max_ksize, isize); ksize ++)
                {
                        for (int t = 0; t < n_tests; t ++)
                        {
                                test::test_conv3d<ltensor_t>(isize, idims, ksize, odims);
                                test::test_conv3d<htensor_t>(isize, idims, ksize, odims);
                        }
                }
        }
}

