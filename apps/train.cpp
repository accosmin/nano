#include "loss.h"
#include "task.h"
#include "model.h"
#include "io/io.h"
#include <fstream>
#include "trainer.h"
#include "enhancer.h"
#include "checkpoint.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "text/filesystem.h"

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
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_fold = cmdline.get<size_t>("fold");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_trainer = cmdline.get<string_t>("trainer");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_enhancer = cmdline.get<string_t>("enhancer");
        const auto cmd_basepath = cmdline.get<string_t>("basepath");
        const auto cmd_threads = cmdline.get<size_t>("threads");

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

        // load enhancer
        checkpoint.step(strcat("load enhancer configuration from <", cmd_enhancer, ">"));
        checkpoint.critical(load_json(cmd_enhancer, "enhancer", params, id));

        renhancer_t enhancer;
        checkpoint.step(strcat("search enhancer <", id, ">"));
        checkpoint.critical((enhancer = get_enhancers().get(id)) != nullptr);

        enhancer->config(params);

        // load trainer
        checkpoint.step(strcat("load trainer configuration from <", cmd_trainer, ">"));
        checkpoint.critical(load_json(cmd_trainer, "trainer", params, id));

        rtrainer_t trainer;
        checkpoint.step(strcat("search trainer <", id, ">"));
        checkpoint.critical((trainer = get_trainers().get(id)) != nullptr);

        trainer->config(params);

        // load model
        checkpoint.step(strcat("load model configuration from <", cmd_model, ">"));
        checkpoint.critical(load_string(cmd_model, params));

        model_t model;
        checkpoint.step("configure model");
        checkpoint.measure(model.config(params) && model.resize(task->idims(), task->odims()));

        model.random();
        model.describe();
        assert(model == *task);

        // train model
        accumulator_t acc(model, *loss);
        acc.threads(cmd_threads);

        trainer_result_t result;
        checkpoint.step("train model");
        checkpoint.measure((result = trainer->train(*enhancer, *task, cmd_fold, acc)) == true);

        model.params(result.optimum_params());

        // save the model
        checkpoint.step("save model");
        checkpoint.critical(
                model.save(cmd_basepath + ".model") &&
                save(cmd_basepath + ".state", result.optimum_states()) &&
                save(cmd_basepath + ".probe", acc.probes()));

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
