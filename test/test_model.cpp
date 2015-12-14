#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_model"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "math/epsilon.hpp"
#include "cortex/evaluate.h"
#include "text/to_string.hpp"
#include "cortex/tasks/task_charset.h"
#include <cstdio>

BOOST_AUTO_TEST_CASE(test_model)
{
        cortex::init();

        using namespace cortex;

        charset_task_t task(charset::numeric, 16, 16, color_mode::luma, 128);
        BOOST_CHECK_EQUAL(task.load(""), true);

        const size_t n_tests = 5;

        const string_t lmodel0;
        const string_t lmodel1 = lmodel0 + "linear:dims=10;act-snorm;";
        const string_t lmodel2 = lmodel1 + "linear:dims=10;act-snorm;";
        const string_t lmodel3 = lmodel2 + "linear:dims=10;act-snorm;";
        const string_t lmodel4 = lmodel3 + "linear:dims=10;act-snorm;";
        const string_t lmodel5 = lmodel4 + "linear:dims=10;act-snorm;";
        
        string_t cmodel;
        cmodel = cmodel + "conv:dims=8,rows=7,cols=7;pool-max;act-snorm;";
        cmodel = cmodel + "conv:dims=8,rows=5,cols=5;act-snorm;";

        const string_t outlayer = "linear:dims=" + text::to_string(task.osize()) + ";";

        strings_t cmd_networks =
        {
                lmodel0 + outlayer,
                lmodel1 + outlayer,
                lmodel2 + outlayer,
                lmodel3 + outlayer,
                lmodel4 + outlayer,
                lmodel5 + outlayer,

                cmodel + outlayer
        };

        const rloss_t loss = cortex::get_losses().get("logistic");
        BOOST_CHECK_EQUAL(loss.operator bool(), true);

        for (const string_t& cmd_network : cmd_networks)
        {
                // create feed-forward network
                const rmodel_t model = cortex::get_models().get("forward-network", cmd_network);
                BOOST_CHECK_EQUAL(model.operator bool(), true);
                BOOST_CHECK_EQUAL(model->resize(task, false), true);
                BOOST_CHECK_EQUAL(model->irows(), task.irows());
                BOOST_CHECK_EQUAL(model->icols(), task.icols());
                BOOST_CHECK_EQUAL(model->osize(), task.osize());
                BOOST_CHECK_EQUAL(model->color(), task.color());

                // test random networks
                for (size_t t = 0; t < n_tests; ++ t)
                {
                        model->random_params();

                        const fold_t fold = {0, protocol::test};

                        const string_t header = "test [" + text::to_string(t + 1) + "/" + text::to_string(n_tests) + "] ";
                        const string_t path = "./test_model.test";

                        // test error & parameters before saving
                        scalar_t lvalue_before, lerror_before;
                        const size_t lcount_before = cortex::evaluate(task, fold, *loss, *model, lvalue_before, lerror_before);

                        vector_t params(model->psize());
                        BOOST_CHECK(model->save_params(params));

                        //
                        BOOST_CHECK_EQUAL(model->save(path), true);
                        model->zero_params();
                        BOOST_CHECK_EQUAL(model->load(path), true);
                        //

                        // test error & parameters after loading
                        scalar_t lvalue_after, lerror_after;
                        const size_t lcount_after = cortex::evaluate(task, fold, *loss, *model, lvalue_after, lerror_after);

                        vector_t xparams(model->psize());
                        BOOST_CHECK(model->save_params(xparams));

                        // check
                        BOOST_CHECK_EQUAL(lcount_before, lcount_after);
                        BOOST_CHECK_LE(math::abs(lvalue_before - lvalue_after), math::epsilon0<scalar_t>());
                        BOOST_CHECK_LE(math::abs(lerror_before - lerror_after), math::epsilon0<scalar_t>());

                        BOOST_CHECK_EQUAL(params.size(), xparams.size());
                        BOOST_CHECK_LE((params - xparams).lpNorm<Eigen::Infinity>(), math::epsilon0<scalar_t>());

                        // cleanup
                        std::remove(path.c_str());
                }
        }
}
