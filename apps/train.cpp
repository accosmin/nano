#include "loss.h"
#include "task.h"
#include "model.h"
#include "io/io.h"
#include <fstream>
#include "trainer.h"
#include "enhancer.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "text/filesystem.h"
#include "measure_and_log.h"

using namespace nano;

typename <typename tfactory, typename trobject = typename tfactory::trobject>
static trobject load_object(const tfactory& factory, const cmdline_t& cmdline, const char* name, string_t& json)
{
        const auto json_path = cmdline.get<string_t>(name);

        measure_critical_and_log(
                [&] () { return io::load_string(json_path, json); },
                strcat("load json <", json_path, ">"));

        string_t id;
        measure_critical_and_log(
                [&] () { json_reader_t(json).object(name, id); return !id.empty(); },
                strcat("load json name <", name, ">"));

        trobject object;
        measure_critical_and_log(
                [&] () { return (object = factory.get(id)) != nullptr; },
                strcat("load ", name, " <", id, ">"));

        return object;
}

static bool save(const string_t& path, const probes_t& probes)
{
        table_t table;
        table.header() << "name" << "#flops" << "gflop/s" << "min[us]" << "avg[us]" << "max[us]";

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

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("train a model");
        cmdline.add("", "task",         join(get_tasks().ids()) + " (.json)");
        cmdline.add("", "fold",         "fold index to use for training", "0");
        cmdline.add("", "model",        "model configuration (.json)");
        cmdline.add("", "trainer",      join(get_trainers().ids()) + " (.json)");
        cmdline.add("", "loss",         join(get_losses().ids()));
        cmdline.add("", "enhancer",     join(get_enhancers().ids()) + " (.json)");
        cmdline.add("", "basepath",     "basepath where to save results (e.g. model, logs, history)");
        cmdline.add("", "threads",      "number of threads to use", physical_cpus());

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_fold = cmdline.get<size_t>("fold");
        const auto cmd_model_params = cmdline.get<string_t>("model-params");
        const auto cmd_model_file = cmdline.get<string_t>("model-file");
        const auto cmd_state_file = dirname(cmd_model_file) + stem(cmd_model_file) + ".state";
        const auto cmd_probe_file = dirname(cmd_model_file) + stem(cmd_model_file) + ".probes";
        const auto cmd_threads = cmdline.get<size_t>("threads");

        // load task
        string_t task_params;
        const auto task = load_object(get_tasks(), cmdline, "task", task_params);
        task->config(cmd_task_params);
        measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");
        describe(*task, cmd_task);

        // load loss
        string_t loss_params;
        const auto loss = load_object(get_losses(), cmdline, "loss", loss_params);

        // load enhancer
        string_t enhancer_params;
        const auto enhancer = load_object(get_enhancers(), cmdline, "enhancer", enhancer_params);
        enhancer->config(enhancer_params);

        // load model
        string_t
        model_t model;
        measure_critical_and_log(
                [&] () { return model.config(cmd_model_params) && model.resize(task->idims(), task->odims()); },
                "configure model");
        model.random();
        model.describe();

        if (model != *task)
        {
                log_error() << "mis-matching model and task!";
                return EXIT_FAILURE;
        }

        // create trainer
        const auto trainer = get_trainers().get(cmd_trainer);
        trainer->config(cmd_trainer_params);

        // train model
        accumulator_t acc(model, *loss);
        acc.threads(cmd_threads);

        // todo: setup the minibatch size
        // todo: add --probes && --probes-detailed to print computation statistics

        trainer_result_t result;
        measure_critical_and_log([&] ()
                {
                        result = trainer->train(*enhancer, *task, cmd_task_fold, acc);
                        return result.valid();
                },
                "train model");

        if (result.valid())
        {
                model.params(result.optimum_params());
        }

        // save the model & its optimization history
        measure_critical_and_log(
                [&] () { return model.save(cmd_model_file); },
                "save model to <" + cmd_model_file + ">");

        measure_critical_and_log(
                [&] () { return save(cmd_state_file, result.optimum_states()); },
                "save state to <" + cmd_state_file + ">");

        measure_critical_and_log(
                [&] () { return save(cmd_probe_file, acc.probes()); },
                "save probes to <" + cmd_probe_file + ">");

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
