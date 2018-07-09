#include "utest.h"
#include "tensor.h"
#include "core/epsilon.h"
#include "core/momentum.h"
#include "tensor/momentum.h"

using namespace nano;

NANO_BEGIN_MODULE(test_momentum)

NANO_CASE(vector)
{
        for (const auto momentum : make_scalars(0.1, 0.5, 0.9))
        {
                const auto dims = 13;
                const auto range = 98;

                momentum_t<vector_t> mom00(momentum, dims);
                momentum_t<vector_t> mom01(momentum, dims);
                momentum_t<vector_t> mom10(1 - momentum, dims);
                momentum_t<vector_t> mom11(1 - momentum, dims);

                const auto epsilon = epsilon1<scalar_t>();
                for (auto i = 1; i <= range; ++ i)
                {
                        const auto base00 = vector_t::Constant(dims, momentum);
                        const auto base01 = vector_t::Constant(dims, 1 - momentum);
                        const auto base10 = vector_t::Constant(dims, momentum);
                        const auto base11 = vector_t::Constant(dims, 1 - momentum);

                        mom00.update(base00);
                        mom01.update(base01);
                        mom10.update(base10);
                        mom11.update(base11);

                        NANO_CHECK_EIGEN_CLOSE(mom00.value(), base00, epsilon);
                        NANO_CHECK_EIGEN_CLOSE(mom01.value(), base01, epsilon);
                        NANO_CHECK_EIGEN_CLOSE(mom10.value(), base10, epsilon);
                        NANO_CHECK_EIGEN_CLOSE(mom11.value(), base11, epsilon);
                }
        }
}

NANO_CASE(scalar)
{
        for (const auto momentum : make_scalars(0.1, 0.5, 0.9))
        {
                const auto range = 98;

                momentum1_t<scalar_t> mom00(momentum);
                momentum1_t<scalar_t> mom01(momentum);
                momentum1_t<scalar_t> mom10(1 - momentum);
                momentum1_t<scalar_t> mom11(1 - momentum);

                const auto epsilon = epsilon1<scalar_t>();
                for (auto i = 1; i <= range; ++ i)
                {
                        const auto base00 = momentum;
                        const auto base01 = 1 - momentum;
                        const auto base10 = momentum;
                        const auto base11 = 1 - momentum;

                        mom00.update(base00);
                        mom01.update(base01);
                        mom10.update(base10);
                        mom11.update(base11);

                        NANO_CHECK_CLOSE(mom00.value(), base00, epsilon);
                        NANO_CHECK_CLOSE(mom01.value(), base01, epsilon);
                        NANO_CHECK_CLOSE(mom10.value(), base10, epsilon);
                        NANO_CHECK_CLOSE(mom11.value(), base11, epsilon);
                }
        }
}

NANO_END_MODULE()
