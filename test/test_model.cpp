#include "nano.h"
#include "utest.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "math/numeric.h"
#include "layers/make_layers.h"

NANO_BEGIN_MODULE(test_model)

NANO_CASE(evaluate)
{
        using namespace nano;

        const auto task = get_tasks().get("synth-charset",
                to_params("type", "digit", "color", "luma", "irows", 16, "icols", 16, "count", 128));

        NANO_CHECK(task->load());

        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(10);
        const string_t mlp2 = mlp1 + make_affine_layer(10);
        const string_t mlp3 = mlp2 + make_affine_layer(10);
        const string_t mlp4 = mlp3 + make_affine_layer(10);
        const string_t mlp5 = mlp4 + make_affine_layer(10);

        const string_t convnet =
                make_conv_layer(8, 7, 7, 1) +
                make_conv_layer(8, 5, 5, 1);

        const string_t outlayer = make_output_layer(task->odims());

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

        const auto loss = nano::get_losses().get("logistic");
        const auto criterion = nano::get_criteria().get("avg");

        for (const string_t& cmd_network : cmd_networks)
        {
                // create feed-forward network
                const auto model = nano::get_models().get("forward-network", cmd_network);
                NANO_CHECK_EQUAL(model->resize(*task), true);
                NANO_CHECK_EQUAL(model->idims(), task->idims());
                NANO_CHECK_EQUAL(model->odims(), task->odims());

                // test random networks
                for (size_t t = 0; t < 5; ++ t)
                {
                        model->random();

                        const fold_t fold = {0, protocol::test};

                        const string_t path = "./test_model.test";

                        // test error & parameters before saving
                        accumulator_t bacc(*model, *loss, *criterion);
                        bacc.mode(criterion_t::type::value);
                        bacc.update(*task, fold);
                        const auto lvalue_before = bacc.value();
                        const auto lerror_before = bacc.estats().avg();
                        const auto lcount_before = bacc.count();

                        vector_t params(model->psize());
                        NANO_CHECK(model->save(params));

                        //
                        NANO_CHECK_EQUAL(model->save(path), true);
                        model->random();
                        NANO_CHECK_EQUAL(model->load(path), true);
                        //

                        // test error & parameters after loading
                        accumulator_t aacc(*model, *loss, *criterion);
                        aacc.mode(criterion_t::type::value);
                        aacc.update(*task, fold);
                        const auto lvalue_after = aacc.value();
                        const auto lerror_after = aacc.estats().avg();
                        const auto lcount_after = aacc.count();

                        vector_t xparams(model->psize());
                        NANO_CHECK(model->save(xparams));

                        // check
                        NANO_CHECK_EQUAL(lcount_before, lcount_after);
                        NANO_CHECK_CLOSE(lvalue_before, lvalue_after, nano::epsilon0<scalar_t>());
                        NANO_CHECK_CLOSE(lerror_before, lerror_after, nano::epsilon0<scalar_t>());

                        NANO_REQUIRE_EQUAL(params.size(), xparams.size());
                        NANO_CHECK_EIGEN_CLOSE(params, xparams, nano::epsilon0<scalar_t>());

                        // cleanup
                        std::remove(path.c_str());
                }
        }
}

NANO_END_MODULE()
