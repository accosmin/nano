#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_model"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "math/epsilon.hpp"
#include "cortex/evaluate.h"
#include "text/to_string.hpp"
#include "cortex/layers/make_layers.h"
#include <cstdio>

BOOST_AUTO_TEST_CASE(test_model)
{
        cortex::init();

        using namespace cortex;

        const auto task = cortex::get_tasks().get("random", "dims=2,rows=16,cols=16,color=luma,size=128");
        BOOST_CHECK_EQUAL(task->load(""), true);

        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(10);
        const string_t mlp2 = mlp1 + make_affine_layer(10);
        const string_t mlp3 = mlp2 + make_affine_layer(10);
        const string_t mlp4 = mlp3 + make_affine_layer(10);
        const string_t mlp5 = mlp4 + make_affine_layer(10);
        
        const string_t convnet =
                make_conv_pool_layer(8, 7, 7) +
                make_conv_layer(8, 5, 5);

        const string_t outlayer = make_output_layer(task->osize());

        strings_t cmd_networks =
        {
                mlp0 + outlayer,
                mlp1 + outlayer,
                mlp2 + outlayer,
                mlp3 + outlayer,
                mlp4 + outlayer,
                mlp5 + outlayer,

                convnet + outlayer
        };

        const auto loss = cortex::get_losses().get("logistic");
        const auto criterion = cortex::get_criteria().get("avg");

        for (const string_t& cmd_network : cmd_networks)
        {
                // create feed-forward network
                const auto model = cortex::get_models().get("forward-network", cmd_network);
                BOOST_CHECK_EQUAL(model->resize(*task, false), true);
                BOOST_CHECK_EQUAL(model->irows(), task->irows());
                BOOST_CHECK_EQUAL(model->icols(), task->icols());
                BOOST_CHECK_EQUAL(model->osize(), task->osize());
                BOOST_CHECK_EQUAL(model->color(), task->color());

                // test random networks
                for (size_t t = 0; t < 5; ++ t)
                {
                        model->random_params();

                        const fold_t fold = {0, protocol::test};

                        const string_t path = "./test_model.test";

                        // test error & parameters before saving
                        scalar_t lvalue_before, lerror_before;
                        const size_t lcount_before = cortex::evaluate(*task, fold, *loss, *criterion, *model,
                                                                      lvalue_before, lerror_before);

                        vector_t params(model->psize());
                        BOOST_CHECK(model->save_params(params));

                        //
                        BOOST_CHECK_EQUAL(model->save(path), true);
                        model->zero_params();
                        BOOST_CHECK_EQUAL(model->load(path), true);
                        //

                        // test error & parameters after loading
                        scalar_t lvalue_after, lerror_after;
                        const size_t lcount_after = cortex::evaluate(*task, fold, *loss, *criterion, *model,
                                                                     lvalue_after, lerror_after);

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
