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
        model_t model;
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
                writer.end_array().next();
                writer.name("model").new_array();
                        writer.array("norm", "c5x5", "act1", "c3x3", "act2", "c1x1", "act3").next();
                        writer.array("act3", "aff1", "act4", "aff2", "act5", "aff3");
                writer.end_array().end_object();

                NANO_CHECK(model.config(writer.str()));
        }

        model_t xmodel;
        {
                NANO_CHECK(add_node(xmodel, "norm", norm3d_node_name(), config_norm3d_node, norm_type::plane));
                NANO_CHECK(add_node(xmodel, "c5x5", conv3d_node_name(), config_conv3d_node, 128, 5, 5, 1, 1, 1));
                NANO_CHECK(add_node(xmodel, "act1", "act-snorm", config_empty_node));
                NANO_CHECK(add_node(xmodel, "c3x3", conv3d_node_name(), config_conv3d_node, 128, 3, 3, 1, 1, 1));
                NANO_CHECK(add_node(xmodel, "act2", "act-snorm", config_empty_node));
                NANO_CHECK(add_node(xmodel, "c1x1", conv3d_node_name(), config_conv3d_node, 128, 1, 1, 1, 1, 1));
                NANO_CHECK(add_node(xmodel, "act3", "act-snorm", config_empty_node));
                NANO_CHECK(add_node(xmodel, "aff1", affine_node_name(), config_affine_node, 128, 1, 1));
                NANO_CHECK(add_node(xmodel, "act4", "act-snorm", config_empty_node));
                NANO_CHECK(add_node(xmodel, "aff2", affine_node_name(), config_affine_node, 128, 1, 1));
                NANO_CHECK(add_node(xmodel, "act5", "act-snorm", config_empty_node));
                NANO_CHECK(add_node(xmodel, "aff3", affine_node_name(), config_affine_node, 10, 1, 1));

                NANO_CHECK(!add_node(xmodel, "xxx", "this node type does not exist", config_empty_node));

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

        json_writer_t writer, xwriter;
        model.config(writer);
        xmodel.config(xwriter);

        NANO_CHECK_EQUAL(writer.str(), xwriter.str());
}

NANO_CASE(graph_empty)
{
        model_t model;
        NANO_CHECK(!model.done());
}

NANO_CASE(graph_many_sources)
{
        json_writer_t writer;
        writer.new_object().name("nodes").new_array();
                add_activation_node(writer, "node1", "act-snorm").next();
                add_activation_node(writer, "node2", "act-snorm").next();
                add_activation_node(writer, "node3", "act-snorm").next();
                add_activation_node(writer, "node4", "act-snorm").next();
                add_activation_node(writer, "node5", "act-snorm").next();
                add_activation_node(writer, "node6", "act-snorm").next();
        writer.end_array().next();

        writer.name("model").new_array();
                writer.array("node1", "node4", "node5", "node6").next();
                writer.array("node2", "node4", "node5", "node6").next();
                writer.array("node3", "node4", "node5", "node6").next();
        writer.end_array().end_object();

        model_t model;
        NANO_CHECK(!model.config(writer.str()));
}

NANO_CASE(graph_many_sinks)
{
        json_writer_t writer;
        writer.new_object().name("nodes").new_array();
                add_activation_node(writer, "node1", "act-snorm").next();
                add_activation_node(writer, "node2", "act-snorm").next();
                add_activation_node(writer, "node3", "act-snorm").next();
                add_activation_node(writer, "node4", "act-snorm").next();
                add_activation_node(writer, "node5", "act-snorm").next();
                add_activation_node(writer, "node6", "act-snorm").next();
        writer.end_array().next();

        writer.name("model").new_array();
                writer.array("node1", "node4", "node5", "node6").next();
                writer.array("node1", "node2", "node3");
        writer.end_array().end_object();

        model_t model;
        NANO_CHECK(!model.config(writer.str()));
}

NANO_CASE(graph_cyclic)
{
        json_writer_t writer;
        writer.new_object().name("nodes").new_array();
                add_activation_node(writer, "node1", "act-snorm").next();
                add_activation_node(writer, "node2", "act-snorm").next();
                add_activation_node(writer, "node3", "act-snorm").next();
                add_activation_node(writer, "node4", "act-snorm").next();
                add_activation_node(writer, "node5", "act-snorm").next();
                add_activation_node(writer, "node6", "act-snorm").next();
        writer.end_array().next();

        writer.name("model").new_array();
                writer.array("node1", "node2", "node3", "node4", "node5", "node6").next();
                writer.array("node5", "node2");
        writer.end_array().end_object();

        model_t model;
        NANO_CHECK(!model.config(writer.str()));
}

NANO_CASE(evaluate)
{
        // setup synthetic task
        json_writer_t writer;
        writer.object("type", "digit", "color", "luma", "irows", 16, "icols", 16, "count", 128);

        const auto task = get_tasks().get("synth-charset");
        task->config(writer.str());
        NANO_CHECK(task->load());

        const auto omaps = std::get<0>(task->odims());
        const auto orows = std::get<1>(task->odims());
        const auto ocols = std::get<2>(task->odims());

        // create & configure feed-forward network
        model_t model;
        NANO_CHECK(add_node(model, "norm", norm3d_node_name(), config_norm3d_node, norm_type::plane));
        NANO_CHECK(add_node(model, "c5x5", conv3d_node_name(), config_conv3d_node, 8, 5, 5, 1, 1, 1));
        NANO_CHECK(add_node(model, "act1", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "c3x3", conv3d_node_name(), config_conv3d_node, 8, 3, 3, 1, 1, 1));
        NANO_CHECK(add_node(model, "act2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "c1x1", conv3d_node_name(), config_conv3d_node, 8, 1, 1, 1, 1, 1));
        NANO_CHECK(add_node(model, "act3", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "aff1", affine_node_name(), config_affine_node, 8, 1, 1));
        NANO_CHECK(add_node(model, "act4", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "aff2", affine_node_name(), config_affine_node, 8, 1, 1));
        NANO_CHECK(add_node(model, "act5", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "aff3", affine_node_name(), config_affine_node, omaps, orows, ocols));

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
