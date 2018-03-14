#include "loss.h"
#include "task.h"
#include "utest.h"
#include "builder.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "math/numeric.h"

using namespace nano;

NANO_BEGIN_MODULE(test_model)

NANO_CASE(config)
{
        model_t umodel;
        {
                json_t json;
                json["nodes"] =
                {
                        config_norm3d_node("norm", norm_type::plane),
                        config_conv3d_node("c5x5", 128, 5, 5, 1, 1, 1),
                        config_activation_node("act1", "act-snorm"),
                        config_conv3d_node("c3x3", 128, 3, 3, 1, 1, 1),
                        config_activation_node("act2", "act-snorm"),
                        config_conv3d_node("c1x1", 128, 1, 1, 1, 1, 1),
                        config_activation_node("act3", "act-snorm"),
                        config_affine_node("aff1", 128, 1, 1),
                        config_activation_node("act4", "act-snorm"),
                        config_affine_node("aff2", 128, 1, 1),
                        config_activation_node("act5", "act-snorm"),
                        config_affine_node("aff3", 10, 1, 1)
                };
                json["model"] =
                {
                        { "norm", "c5x5", "act1", "c3x3", "act2", "c1x1", "act3" },
                        { "act3", "aff1", "act4", "aff2", "act5", "aff3" }
                };

                NANO_CHECK(umodel.from_json(json));
        }

        model_t xmodel;
        {
                NANO_CHECK(xmodel.add(config_norm3d_node("norm", norm_type::plane)));
                NANO_CHECK(xmodel.add(config_conv3d_node("c5x5", 128, 5, 5, 1, 1, 1)));
                NANO_CHECK(xmodel.add(config_activation_node("act1", "act-snorm")));
                NANO_CHECK(xmodel.add(config_conv3d_node("c3x3", 128, 3, 3, 1, 1, 1)));
                NANO_CHECK(xmodel.add(config_activation_node("act2", "act-snorm")));
                NANO_CHECK(xmodel.add(config_conv3d_node("c1x1", 128, 1, 1, 1, 1, 1)));
                NANO_CHECK(xmodel.add(config_activation_node("act3", "act-snorm")));
                NANO_CHECK(xmodel.add(config_affine_node("aff1", 128, 1, 1)));
                NANO_CHECK(xmodel.add(config_activation_node("act4", "act-snorm")));
                NANO_CHECK(xmodel.add(config_affine_node("aff2", 128, 1, 1)));
                NANO_CHECK(xmodel.add(config_activation_node("act5", "act-snorm")));
                NANO_CHECK(xmodel.add(config_affine_node("aff3", 10, 1, 1)));

                NANO_CHECK(!xmodel.add(config_activation_node("xxx", "act-this activation type does not exist")));

                NANO_CHECK(xmodel.connect("norm", "c5x5"));
                NANO_CHECK(xmodel.connect("c5x5", "act1"));
                NANO_CHECK(xmodel.connect("act1", "c3x3"));
                NANO_CHECK(xmodel.connect("c3x3", "act2"));
                NANO_CHECK(xmodel.connect("act2", "c1x1"));
                NANO_CHECK(xmodel.connect("c1x1", "act3"));
                NANO_CHECK(xmodel.connect("act3", "aff1"));
                NANO_CHECK(xmodel.connect("aff1", "act4"));
                NANO_CHECK(xmodel.connect("act4", "aff2"));
                NANO_CHECK(xmodel.connect("aff2", "act5"));
                NANO_CHECK(xmodel.connect("act5", "aff3"));

                NANO_CHECK(!xmodel.connect("unknown-node-name", "what?!"));

                NANO_CHECK(xmodel.done());
        }

        model_t ymodel;
        {
                NANO_CHECK(ymodel.add(config_norm3d_node("norm", norm_type::plane)));
                NANO_CHECK(ymodel.add(config_conv3d_node("c5x5", 128, 5, 5, 1, 1, 1)));
                NANO_CHECK(ymodel.add(config_activation_node("act1", "act-snorm")));
                NANO_CHECK(ymodel.add(config_conv3d_node("c3x3", 128, 3, 3, 1, 1, 1)));
                NANO_CHECK(ymodel.add(config_activation_node("act2", "act-snorm")));
                NANO_CHECK(ymodel.add(config_conv3d_node("c1x1", 128, 1, 1, 1, 1, 1)));
                NANO_CHECK(ymodel.add(config_activation_node("act3", "act-snorm")));
                NANO_CHECK(ymodel.add(config_affine_node("aff1", 128, 1, 1)));
                NANO_CHECK(ymodel.add(config_activation_node("act4", "act-snorm")));
                NANO_CHECK(ymodel.add(config_affine_node("aff2", 128, 1, 1)));
                NANO_CHECK(ymodel.add(config_activation_node("act5", "act-snorm")));
                NANO_CHECK(ymodel.add(config_affine_node("aff3", 10, 1, 1)));

                NANO_CHECK(ymodel.connect(strings_t{"norm", "c5x5"}));
                NANO_CHECK(ymodel.connect(strings_t{"c5x5", "act1", "c3x3", "act2", "c1x1", "act3", "aff1", "act4", "aff2", "act5", "aff3"}));

                NANO_CHECK(ymodel.done());
        }

        NANO_CHECK_EQUAL(umodel.to_json().dump(), xmodel.to_json().dump());
        NANO_CHECK_EQUAL(umodel.to_json().dump(), ymodel.to_json().dump());
}

NANO_CASE(config_model_before_nodes)
{
        const string_t config = R"XXX(
{
        "model": [],
        "nodes": [
                {"name": "output", "type": "affine", "omaps": 8, "orows": 1, "ocols": 1}
        ]
})XXX";

        model_t model;
        NANO_CHECK(model.from_json(json_t::parse(config)));
}

NANO_CASE(config_nodes_before_model)
{
        const string_t config = R"XXX(
{
        "nodes": [
                {"name": "output", "type": "affine", "omaps": 8, "orows": 1, "ocols": 1}
        ],
        "model": []
})XXX";

        model_t model;
        NANO_CHECK(model.from_json(json_t::parse(config)));
}

NANO_CASE(graph_empty)
{
        model_t model;
        NANO_CHECK(!model.done());
}

NANO_CASE(graph_many_sinks)
{
        json_t json;
        json["nodes"] =
        {
                config_activation_node("node1", "act-snorm"),
                config_activation_node("node2", "act-snorm"),
                config_activation_node("node3", "act-snorm"),
                config_activation_node("node4", "act-snorm"),
                config_activation_node("node5", "act-snorm"),
                config_activation_node("node6", "act-snorm")
        };
        json["model"] =
        {
                { "node1", "node4", "node5", "node6" },
                { "node1", "node2", "node3" }
        };

        model_t model;
        NANO_CHECK(!model.from_json(json));
}

NANO_CASE(graph_cyclic)
{
        json_t json;
        json["nodes"] =
        {
                config_activation_node("node1", "act-snorm"),
                config_activation_node("node2", "act-snorm"),
                config_activation_node("node3", "act-snorm"),
                config_activation_node("node4", "act-snorm"),
                config_activation_node("node5", "act-snorm"),
                config_activation_node("node6", "act-snorm")
        };
        json["model"] =
        {
                { "node1", "node2", "node3", "node4", "node5", "node6" },
                { "node5", "node2" }
        };

        model_t model;
        NANO_CHECK(!model.from_json(json));
}

NANO_CASE(evaluate)
{
        // setup synthetic task
        const auto task = get_tasks().get("synth-peak2d");
        NANO_REQUIRE(task);
        task->from_json(to_json("irows", 16, "icols", 16, "count", 128));
        NANO_CHECK(task->load());

        const auto omaps = std::get<0>(task->odims());
        const auto orows = std::get<1>(task->odims());
        const auto ocols = std::get<2>(task->odims());

        // create & configure feed-forward network
        model_t model;
        NANO_CHECK(model.add(config_norm3d_node("norm", norm_type::plane)));
        NANO_CHECK(model.add(config_conv3d_node("c5x5", 8, 5, 5, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("act1", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("c3x3", 8, 3, 3, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("act2", "act-snorm")));
        NANO_CHECK(model.add(config_conv3d_node("c1x1", 8, 1, 1, 1, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("act3", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("aff1", 8, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("act4", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("aff2", 8, 1, 1)));
        NANO_CHECK(model.add(config_activation_node("act5", "act-snorm")));
        NANO_CHECK(model.add(config_affine_node("aff3", omaps, orows, ocols)));

        NANO_CHECK(model.connect("norm", "c5x5", "act1", "c3x3", "act2", "c1x1", "act3"));
        NANO_CHECK(model.connect("act3", "aff1", "act4", "aff2", "act5", "aff3"));

        NANO_CHECK(model.done());
        NANO_REQUIRE(model.resize(task->idims(), task->odims()));
        NANO_CHECK_EQUAL(model.idims(), task->idims());
        NANO_CHECK_EQUAL(model.odims(), task->odims());

        model.random();

        const auto fold = fold_t{0, protocol::test};
        const auto path = string_t("./test_model.test");
        const auto loss = get_losses().get("s-logistic");

        // test error & parameters before saving
        accumulator_t bacc(model, *loss);
        bacc.mode(accumulator_t::type::value);
        bacc.update(*task, fold);
        const auto lvalue_before = bacc.vstats().avg();
        const auto lerror_before = bacc.estats().avg();
        const auto lcount_before = bacc.vstats().count();

        const auto params = model.params();
        NANO_CHECK_EQUAL(params.size(), model.psize());

        //
        NANO_REQUIRE(model.save(path));
        model.random();
        NANO_REQUIRE(model.load(path));
        NANO_CHECK_EQUAL(model.idims(), task->idims());
        NANO_CHECK_EQUAL(model.odims(), task->odims());
        //

        // test error & parameters after loading
        accumulator_t aacc(model, *loss);
        aacc.mode(accumulator_t::type::value);
        aacc.update(*task, fold);
        const auto lvalue_after = aacc.vstats().avg();
        const auto lerror_after = aacc.estats().avg();
        const auto lcount_after = aacc.vstats().count();

        const auto xparams = model.params();
        NANO_CHECK_EQUAL(xparams.size(), model.psize());

        // the outputs & parameters should match before & after serialization to disk
        NANO_CHECK_EQUAL(lcount_before, lcount_after);
        NANO_CHECK_CLOSE(lvalue_before, lvalue_after, epsilon0<scalar_t>());
        NANO_CHECK_CLOSE(lerror_before, lerror_after, epsilon0<scalar_t>());

        NANO_REQUIRE_EQUAL(params.size(), xparams.size());
        NANO_CHECK_EIGEN_CLOSE(params, xparams, epsilon0<scalar_t>());

        // cleanup
        std::remove(path.c_str());
}

NANO_END_MODULE()
