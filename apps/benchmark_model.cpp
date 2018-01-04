#include "loss.h"
#include "task.h"
#include "model.h"
#include "io/io.h"
#include "checkpoint.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"

using namespace nano;

static bool load_json(const string_t& path, const char* name, string_t& json, string_t& id)
{
        if (!load_string(path, json) || json.empty())
        {
                return false;
        }

        json_reader_t(json).object(name, id);
        return !id.empty();
}

static bool save(const string_t& path, const probes_t& probes)
{
        table_t table;
        table.header() << "name" << "#flops" << "gflop/s" << "min[us]" << "avg[us]" << "max[us]";
        table.delim();

        for (const auto& probe : probes)
        {
                auto& row = table.append();
                row << probe.fullname() << probe.flops();
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

        std::ofstream os(path.c_str());
        return os.is_open() && (os << table);
}

static void append(table_t& table, const string_t& name, const tensor_size_t params, const size_t minibatch,
        const probes_t& probes, const bool detailed)
{
        for (const auto& probe : probes)
        {
                if (!starts_with(probe.fullname(), "network") && !detailed)
                {
                        continue;
                }

                auto& row = table.append();
                row << (name + " " + probe.fullname()) << params << probe.flops() << minibatch;
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

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark a model");
        cmdline.add("", "task",         join(get_tasks().ids()) + " (.json)");
        cmdline.add("", "model",        "model configuration (.json)");
        cmdline.add("", "loss",         join(get_losses().ids()) + " (.json)");
        cmdline.add("", "threads",      "number of threads to use", physical_cpus());
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
        const auto cmd_threads = cmdline.get<size_t>("threads");
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
        string_t params, id;

        // load task
        checkpoint.step(strcat("load task configuration from <", cmd_task, ">"));
        checkpoint.critical(load_json(cmd_task, "task", params, id));

        rtask_t task;
        checkpoint.step(strcat("search task <", id, ">"));
        checkpoint.critical((task = get_tasks().get(id)) != nullptr);

        task->config(params);
        checkpoint.step(strcat("load task <", id, ">"));
        checkpoint.measure(task->load());

        describe(*task, id);

        // load loss
        checkpoint.step(strcat("load loss configuration from <", cmd_loss, ">"));
        checkpoint.critical(load_json(cmd_loss, "loss", params, id));

        rloss_t loss;
        checkpoint.step(strcat("search loss <", id, ">"));
        checkpoint.critical((loss = get_losses().get(id)) != nullptr);

        // load model
        checkpoint.step(strcat("load model configuration from <", cmd_model, ">"));
        checkpoint.critical(load_string(cmd_model, params));

        model_t model;
        checkpoint.step("configure model");
        checkpoint.measure(model.config(params) && model.resize(task->idims(), task->odims()));

        model.random();
        model.describe();
        if (model != *task)
        {
                log_error() << "model not compatible with the task!";
                return EXIT_FAILURE;
        }

        // construct tables to compare models
        table_t table;
        table.header() << "network" << "#params" << "#flops" << "batch" << "gflop/s" << "min[us]" << "avg[us]" << "max[us]";
        table.delim();

        // evaluate models
        for (const auto& config : networks)
        {
                const auto cmd_network = config.first;
                const auto cmd_name = config.second;

                // create feed-forward network
                model_t model(cmd_network);
                model.config(task->idims(), task->odims());
                model.random();
                model.describe();

                const auto fold = fold_t{0, protocol::train};
                const auto size = task->size(fold);

                for (size_t count = cmd_min_count; count <= cmd_max_count; count *= 2)
                {
                        // measure processing
                        accumulator_t acc(model, *loss);
                        acc.threads(1);
                        acc.mode((cmd_forward && !cmd_backward) ? accumulator_t::type::value : accumulator_t::type::vgrad);

                        for (size_t i = 0; i + count < size; i += count)
                        {
                                acc.update(*task, fold, i, i + count);
                        }

                        log_info()
                                << "<<< processed [" << size << "] samples using "
                                << cmd_name << " and minibatch of " << count << ".";

                        append(table, cmd_name, model.psize(), count, acc.probes(), cmd_detailed);

                        if (cmd_detailed && count * 2 <= cmd_max_count)
                        {
                                table.delim();
                        }
                }

                const auto last = config == *networks.rbegin();
                if (!cmd_detailed && !last)
                {
                        table.delim();
                }
        }

        // print results
        std::cout << table;

        // OK
        return EXIT_SUCCESS;
}
