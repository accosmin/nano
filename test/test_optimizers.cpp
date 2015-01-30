#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_optimizers"

#include <boost/test/unit_test.hpp>
#include "libnanocv/optimize.h"
#include "libnanocv/util/logger.h"
#include <iostream>

namespace test
{
        using namespace ncv;

        void check(const opt_opsize_t& fn_size,
                   const opt_opfval_t& fn_fval,
                   const opt_opgrad_t& fn_grad)
        {
                const size_t iterations = 8 * 1024;
                const scalar_t epsilon = 1e-6;
                const size_t history = 6;

                const auto optimizers =
                {
                        batch_optimizer::GD,
                        batch_optimizer::CGD,
                        batch_optimizer::LBFGS
                };

                const size_t dims = fn_size();

                vector_t x0(dims);
                x0.setRandom();

                for (batch_optimizer optimizer : optimizers)
                {
                        const opt_state_t state = ncv::minimize(fn_size, fn_fval, fn_grad, nullptr, nullptr, nullptr,
                                                                x0, optimizer, iterations, epsilon, history);

                        log_info() << "dims = " << fn_size()
                                   << ", optimizer = " << text::to_string(optimizer)
                                   << ", fx = " << state.f
                                   << ", gx = " << state.g.lpNorm<Eigen::Infinity>()
                                   << ", evals = " << state.m_n_fvals << "/" << state.m_n_grads;
                }
        }
}

BOOST_AUTO_TEST_CASE(test_optimizers)
{
        using namespace ncv;        

        // Sphere function
        for (size_t dims = 1; dims <= 16; dims *= 2)
        {
                const opt_opsize_t fn_size = [=] ()
                {
                        return dims;
                };

                const opt_opfval_t fn_fval = [=] (const vector_t& x)
                {
                        scalar_t fx = 0;
                        for (size_t i = 0; i < dims; i ++)
                        {
                                fx += x(i) * x(i);
                        }

                        return fx;
                };

                const opt_opgrad_t fn_grad = [=] (const vector_t& x, vector_t& gx)
                {
                        gx.resize(dims);

                        scalar_t fx = 0;
                        for (size_t i = 0; i < dims; i ++)
                        {
                                fx += x(i) * x(i);
                                gx(i) = 2 * x(i);
                        }

                        return fx;
                };

                test::check(fn_size, fn_fval, fn_grad);
        }
}

