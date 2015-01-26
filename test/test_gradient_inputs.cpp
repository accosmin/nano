#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_gradient_inputs"

#include <boost/test/unit_test.hpp>
#include "nanocv.h"
#include "accumulator.h"
#include "util/logger.h"
#include "util/close.hpp"
#include "util/random.hpp"
#include "util/epsilon.hpp"
#include "util/thread_loop.hpp"
#include <set>

namespace test
{
        using namespace ncv;

        thread_pool_t::mutex_t mutex;

        size_t n_checks = 0;
        size_t n_failures1 = 0;
        size_t n_failures2 = 0;
        size_t n_failures3 = 0;

        void test_grad(const string_t& header, const string_t& loss_id, const model_t& model)
        {
                rmodel_t rmodel_inputs = model.clone();

                const size_t n_tests = 64;

                const rloss_t rloss = loss_manager_t::instance().get(loss_id);
                const loss_t& loss = *rloss;

                const size_t psize = model.psize();
                const size_t isize = model.isize();
                const size_t osize = model.osize();

                vector_t params(psize);
                vector_t target(osize);
                tensor_t input(model.idims(), model.irows(), model.icols());

                // optimization problem (wrt parameters & inputs): size
                auto fn_inputs_size = [&] ()
                {
                        return isize;
                };

                // optimization problem (wrt parameters & inputs): function value
                auto fn_inputs_fval = [&] (const vector_t& x)
                {
                        rmodel_inputs->load_params(params);

                        const vector_t output = rmodel_inputs->output(x).vector();

                        return loss.value(target, output);
                };

                // optimization problem (wrt parameters & inputs): function value & gradient
                auto fn_inputs_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        rmodel_inputs->load_params(params);

                        const vector_t output = rmodel_inputs->output(x).vector();

                        gx = rmodel_inputs->ginput(loss.vgrad(target, output)).vector();
                        return loss.value(target, output);
                };

                // construct optimization problem: analytic gradient and finite difference approximation
                const opt_problem_t problem_analytic_inputs(fn_inputs_size, fn_inputs_fval, fn_inputs_grad);
                const opt_problem_t problem_aproxdif_inputs(fn_inputs_size, fn_inputs_fval);

                for (size_t t = 0; t < n_tests; t ++)
                {
                        random_t<scalar_t> prgen(-1.0, +1.0);
                        random_t<scalar_t> irgen(-0.1, +0.1);
                        random_t<size_t> trgen(0, osize - 1);

                        prgen(params.data(), params.data() + psize);
                        target = ncv::class_target(trgen(), osize);
                        irgen(input.data(), input.data() + isize);

                        vector_t analytic_inputs_grad, aproxdif_inputs_grad;

                        problem_analytic_inputs(input.vector(), analytic_inputs_grad);
                        problem_aproxdif_inputs(input.vector(), aproxdif_inputs_grad);

                        const scalar_t delta = (analytic_inputs_grad - aproxdif_inputs_grad).lpNorm<Eigen::Infinity>();

                        // update statistics
                        const thread_pool_t::lock_t lock(test::mutex);

                        n_checks ++;
                        if (!math::close(delta, scalar_t(0), math::epsilon1<scalar_t>()))
                        {
                                n_failures1 ++;
                        }
                        if (!math::close(delta, scalar_t(0), math::epsilon2<scalar_t>()))
                        {
                                n_failures2 ++;
                        }
                        if (!math::close(delta, scalar_t(0), math::epsilon3<scalar_t>()))
                        {
                                n_failures3 ++;

                                BOOST_CHECK_LE(delta, math::epsilon3<scalar_t>());

                                log_error() << header << ": error = " << delta << "/" << math::epsilon3<scalar_t>() << "!";
                        }
                }
        }
}

BOOST_AUTO_TEST_CASE(test_gradient_inputs)
{
        using namespace ncv;

        ncv::init();

        const strings_t conv_layer_ids { "", "conv" };
        const strings_t conv_masks { "25", "50", "100" };
        const strings_t pool_layer_ids { "", "pool-max", "pool-min", "pool-avg" };
        const strings_t full_layer_ids { "", "linear" };
        const strings_t actv_layer_ids { "", "act-unit", "act-tanh", "act-snorm", "act-splus" };
        const strings_t loss_ids = loss_manager_t::instance().ids();

        const color_mode cmd_color = color_mode::luma;
        const size_t cmd_irows = 8;
        const size_t cmd_icols = 8;
        const size_t cmd_outputs = 4;
        const size_t cmd_max_layers = 2;

        // evaluate the analytical gradient vs. the finite difference approximation for various:
        //      * convolution layers
        //      * convolution connection types
        //      * pooling layers
        //      * fully connected layers
        //      * activation layers
        std::set<string_t> descs;
        for (size_t n_layers = 0; n_layers <= cmd_max_layers; n_layers ++)
        {
                for (const string_t& actv_layer_id : actv_layer_ids)
                {
                        for (const string_t& pool_layer_id : pool_layer_ids)
                        {
                                for (const string_t& conv_layer_id : conv_layer_ids)
                                {
                                        for (const string_t& conv_mask : conv_masks)
                                        {
                                                for (const string_t& full_layer_id : full_layer_ids)
                                                {
                                                        string_t desc;

                                                        // convolution part
                                                        for (size_t l = 0; l < n_layers && !conv_layer_id.empty(); l ++)
                                                        {
                                                                random_t<size_t> rgen(2, 3);

                                                                string_t params;
                                                                params += "dims=" + text::to_string(rgen());
                                                                params += (rgen() % 2 == 0) ? ",rows=2,cols=2," : ",rows=3,cols=3,";
                                                                params += "mask=" + conv_mask;

                                                                desc += conv_layer_id + ":" + params + ";";
                                                                if (l == 0)
                                                                {
                                                                        desc += pool_layer_id + ";";
                                                                }
                                                                desc += actv_layer_id + ";";
                                                        }

                                                        // fully-connected part
                                                        for (size_t l = 0; l < n_layers && !full_layer_id.empty(); l ++)
                                                        {
                                                                random_t<size_t> rgen(1, 5);

                                                                string_t params;
                                                                params += "dims=" + text::to_string(rgen());

                                                                desc += full_layer_id + ":" + params + ";";
                                                                desc += actv_layer_id + ";";
                                                        }

                                                        desc += "linear:dims=" + text::to_string(cmd_outputs) + ";";

                                                        descs.insert(desc);
                                                }
                                        }
                                }
                        }
                }
        }

        // test each network
        const strings_t networks(descs.begin(), descs.end());
        ncv::thread_loopi(networks.size(), [&] (size_t i)
        {
                const string_t network = networks[i];

                // create network
                const rmodel_t model = model_manager_t::instance().get("forward-network", network);
                BOOST_CHECK_EQUAL(model.operator bool(), true);
                {
                        const thread_pool_t::lock_t lock(test::mutex);

                        model->resize(cmd_irows, cmd_icols, cmd_outputs, cmd_color, false);
                }

                // check with different loss functions
                for (const string_t& loss_id : loss_ids)
                {
                        test::test_grad("[loss = " + loss_id + "]", loss_id, *model);
                }
        });

        const scalar_t eps1 = math::epsilon1<scalar_t>();
        const scalar_t eps2 = math::epsilon2<scalar_t>();
        const scalar_t eps3 = math::epsilon3<scalar_t>();

        log_info() << "failures: level1 = " << test::n_failures1 << "/" << test::n_checks << ", epsilon = " << eps1;
        log_info() << "failures: level2 = " << test::n_failures2 << "/" << test::n_checks << ", epsilon = " << eps2;
        log_info() << "failures: level3 = " << test::n_failures3 << "/" << test::n_checks << ", epsilon = " << eps3;
}
