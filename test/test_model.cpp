#include "loss.h"
#include "task.h"
#include "layer.h"
#include "model.h"
#include "utest.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "math/numeric.h"
#include "layers/builder.h"

using namespace nano;

NANO_BEGIN_MODULE(test_model)

NANO_CASE(config)
{
        json_writer_t writer;
        writer.new_object().name("nodes").new_array();
                add_norm3d_node(writer, "norm", norm_type::plane).next();
                add_conv3d_node(writer, "c5x5", 128, 5, 5, 1, 1, 1).next();
                add_activation_node(writer, "act1", "act-snorm").next();
                add_conv3d_node(writer, "c3x3", 128, 3, 3, 1, 1, 1).next();
                add_activation_node(writer, "act2", "act-snorm").next();
                add_conv3d_node(writer, "c1x1", 128, 1, 1, 1, 1, 1).next();
                add_activation_node(writer, "act3", "act-snorm").next();
                add_affine_node(writer, "aff1", 128, 1, 1).next();
                add_activation_node(writer, "act4", "act-snorm").next();
                add_affine_node(writer, "aff2", 128, 1, 1).next();
                add_activation_node(writer, "act5", "act-snorm").next();
                add_affine_node(writer, "aff3", 10, 1, 1);
        writer.end_array().next().end_object().next();
        writer.new_object().name("model").new_array();
                writer.array("norm", "c5x5", "act1", "c3x3", "act2", "c1x1", "act3").next();
                writer.array("act3", "aff1", "act4", "aff2", "act5", "aff3");
        writer.end_array().end_object();

        std::cout << writer.str() << std::endl;
}

NANO_CASE(evaluate)
{
        /*
        const auto task = get_tasks().get("synth-charset");
        task->config(json_writer_t().object("type", "digit", "color", "luma", "irows", 16, "icols", 16, "count", 128).get());

        NANO_CHECK(task->load());

        const string_t mlp0;
        const string_t mlp1 = mlp0 + make_affine_layer(10, 1, 1);
        const string_t mlp2 = mlp1 + make_affine_layer(10, 1, 1);
        const string_t mlp3 = mlp2 + make_affine_layer(10, 1, 1);
        const string_t mlp4 = mlp3 + make_affine_layer(10, 1, 1);
        const string_t mlp5 = mlp4 + make_affine_layer(10, 1, 1);

        const string_t convnet =
                make_conv3d_layer(8, 7, 7, 1) +
                make_conv3d_layer(8, 5, 5, 1);

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

        const auto loss = get_losses().get("s-logistic");

        for (const string_t& cmd_network : cmd_networks)
        {
                // create feed-forward network
                model_t model(cmd_network);
                NANO_CHECK(model.config(task->idims(), task->odims()));
                NANO_CHECK_EQUAL(model.idims(), task->idims());
                NANO_CHECK_EQUAL(model.odims(), task->odims());

                // test random networks
                for (size_t t = 0; t < 5; ++ t)
                {
                        model.random();

                        const fold_t fold = {0, protocol::test};

                        const string_t path = "./test_model.test";

                        // test error & parameters before saving
                        accumulator_t bacc(model, *loss);
                        bacc.mode(accumulator_t::type::value);
                        bacc.update(*task, fold);
                        const auto lvalue_before = bacc.vstats().avg();
                        const auto lerror_before = bacc.estats().avg();
                        const auto lcount_before = bacc.vstats().count();

                        const auto params = model.params();

                        //
                        NANO_CHECK_EQUAL(model.save(path), true);
                        model.random();
                        NANO_CHECK_EQUAL(model.load(path), true);
                        //

                        // test error & parameters after loading
                        accumulator_t aacc(model, *loss);
                        aacc.mode(accumulator_t::type::value);
                        aacc.update(*task, fold);
                        const auto lvalue_after = aacc.vstats().avg();
                        const auto lerror_after = aacc.estats().avg();
                        const auto lcount_after = aacc.vstats().count();

                        const auto xparams = model.params();

                        // check
                        NANO_CHECK_EQUAL(lcount_before, lcount_after);
                        NANO_CHECK_CLOSE(lvalue_before, lvalue_after, epsilon0<scalar_t>());
                        NANO_CHECK_CLOSE(lerror_before, lerror_after, epsilon0<scalar_t>());

                        NANO_REQUIRE_EQUAL(params.size(), xparams.size());
                        NANO_CHECK_EIGEN_CLOSE(params, xparams, epsilon0<scalar_t>());

                        // cleanup
                        std::remove(path.c_str());
                }
        }
        */
}

NANO_END_MODULE()
