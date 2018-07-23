#include "loss.h"
#include "task.h"
#include "model.h"
#include "core/io.h"
#include "checkpoint.h"
#include "core/table.h"
#include "accumulator.h"
#include "core/cmdline.h"
#include "core/algorithm.h"
#include <iostream>

using namespace nano;

static bool load_json(const string_t& path, json_t& json)
{
        string_t config;
        const auto ret = load_string(path, config);
        json = json_t::parse(config);
        return ret;
}

static bool load_json(const string_t& path, json_t& json, string_t& id)
{
        return  load_json(path, json) &&
                from_json(json, "type", id);
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark a model");
        cmdline.add("", "task",         join(get_tasks().ids()) + " (.json)");
        cmdline.add("", "model",        "model configuration (.json)");
        cmdline.add("", "loss",         join(get_losses().ids()) + " (.json)");
        cmdline.add("", "forward",      "evaluate the \'forward\' pass (output)");
        cmdline.add("", "backward",     "evaluate the \'backward' pass (gradient)");
        cmdline.add("", "detailed",     "print detailed measurements (e.g. per-layer)");
        cmdline.add("", "min-count",    "minimum number of samples in minibatch [1, 16]",  "1");
        cmdline.add("", "max-count",    "maximum number of samples in minibatch [1, 128]", "16");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_forward = cmdline.has("forward");
        const auto cmd_backward = cmdline.has("backward");
        const auto cmd_detailed = cmdline.has("detailed");
        const auto cmd_min_count = clamp(cmdline.get<size_t>("min-count"), 1, 16);
        const auto cmd_max_count = clamp(cmdline.get<size_t>("max-count"), cmd_min_count, 128);

        if (!cmd_forward && !cmd_backward)
        {
                cmdline.usage();
        }

        checkpoint_t checkpoint;
        json_t json;
        string_t id;

        // load task
        checkpoint.step(strcat("load task configuration from <", cmd_task, ">"));
        checkpoint.critical(load_json(cmd_task, json, id));

        rtask_t task;
        checkpoint.step(strcat("search task <", id, ">"));
        checkpoint.critical((task = get_tasks().get(id)) != nullptr);

        task->from_json(json);
        checkpoint.step(strcat("load task <", id, ">"));
        checkpoint.measure(task->load());

        task->describe(id);

        // load loss
        checkpoint.step(strcat("load loss configuration from <", cmd_loss, ">"));
        checkpoint.critical(load_json(cmd_loss, json, id));

        rloss_t loss;
        checkpoint.step(strcat("search loss <", id, ">"));
        checkpoint.critical((loss = get_losses().get(id)) != nullptr);

        // load model
        checkpoint.step(strcat("load model configuration from <", cmd_model, ">"));
        checkpoint.critical(load_json(cmd_model, json));

        model_t model;
        checkpoint.step("configure model");
        checkpoint.measure(model.from_json(json) && model.resize(task->idims(), task->odims()));

        model.random();
        model.describe();
        if (model != *task)
        {
                log_error() << "model not compatible with the task!";
                return EXIT_FAILURE;
        }

        // benchmark model for different batch sizes
        std::vector<probes_t> batch2probes;
        for (size_t count = cmd_min_count; count <= cmd_max_count; count *= 2)
        {
                const auto fold = fold_t{0, protocol::train};
                const auto size = task->size(fold);

                // measure processing
                accumulator_t acc(model, *loss);
                acc.mode((cmd_forward && !cmd_backward) ? accumulator_t::type::value : accumulator_t::type::vgrad);

                for (size_t i = 0; i + count < size; i += count)
                {
                        acc.update(*task, fold, i, i + count);
                }

                log_info() << "<<< processed [" << size << "] samples using minibatches of size " << count << ".";

                // filter probes
                auto probes = acc.probes();
                probes.erase(
                        std::remove_if(probes.begin(), probes.end(), [&] (const probe_t& probe)
                        {
                                return !starts_with(probe.fullname(), "model") && !cmd_detailed;
                        }),
                        probes.end());

                batch2probes.push_back(probes);
        }

        // print results
        table_t table;
        {
                auto&& header = table.header();
                header << "" << "";
                for (size_t count = cmd_min_count; count <= cmd_max_count; count *= 2)
                {
                        header << colspan(4) << alignment::center << strcat("minibatch x", count);
                }
        }
        table.delim();
        {
                auto&& header = table.header();
                header << "name" << "#flops";
                for (size_t count = cmd_min_count; count <= cmd_max_count; count *= 2)
                {
                        header << "gflop/s" << "min[us]" << "avg[us]" << "max[us]";
                }
        }
        table.delim();
        for (const auto& probe0 : batch2probes[0])
        {
                auto&& row = table.append();
                row << probe0.fullname() << probe0.flops();

                for (const auto& probes : batch2probes)
                {
                        for (const auto& probe : probes)
                        {
                                if (probe.fullname() != probe0.fullname())
                                {
                                        continue;
                                }

                                if (probe.timings().min() < int64_t(1))
                                {
                                        row << "-";
                                }
                                else
                                {
                                        row << probe.gflops();
                                }
                                row << probe.timings().min() << probe.timings().avg() << probe.timings().max();
                        }
                }
        }

        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
