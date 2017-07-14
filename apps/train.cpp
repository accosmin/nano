#include "loss.h"
#include "task.h"
#include "model.h"
#include <fstream>
#include "trainer.h"
#include "enhancer.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include "text/filesystem.h"
#include "measure_and_log.h"

using namespace nano;

bool save(const string_t& path, const probes_t& probes)
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
        if (os.is_open())
        {
                os << table;
                return true;
        }
        else
        {
                return false;
        }
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("train a model");
        cmdline.add("", "task",                 "[" + concatenate(get_tasks().ids()) + "]");
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
        cmdline.add("", "task-fold",            "fold index to use for training", "0");
        cmdline.add("", "model",                "[" + concatenate(get_models().ids()) + "]");
        cmdline.add("", "model-params",         "model parameters (if any)");
        cmdline.add("", "model-file",           "filepath to save the model to");
        cmdline.add("", "trainer",              "[" + concatenate(get_trainers().ids()) + "]");
        cmdline.add("", "trainer-params",       "trainer parameters (if any)");
        cmdline.add("", "loss",                 "[" + concatenate(get_losses().ids()) + "]");
        cmdline.add("", "enhancer",             "[" + concatenate(get_enhancers().ids()) + "]", "default");
        cmdline.add("", "enhancer-params",      "task enhancer parameters (if any)", "-");
        cmdline.add("", "threads",              "number of threads to use", logical_cpus());

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_task_fold = cmdline.get<size_t>("task-fold");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_model_params = cmdline.get<string_t>("model-params");
        const auto cmd_model_file = cmdline.get<string_t>("model-file");
        const auto cmd_state_file = dirname(cmd_model_file) + stem(cmd_model_file) + ".state";
        const auto cmd_probe_file = dirname(cmd_model_file) + stem(cmd_model_file) + ".probes";
        const auto cmd_trainer = cmdline.get<string_t>("trainer");
        const auto cmd_trainer_params = cmdline.get<string_t>("trainer-params");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_enhancer = cmdline.get<string_t>("enhancer");
        const auto cmd_enhancer_params = cmdline.get<string_t>("enhancer-params");
        const auto cmd_threads = cmdline.get<size_t>("threads");

        // create task
        const auto task = get_tasks().get(cmd_task, cmd_task_params);

        // load task data
        measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        // describe task
        describe(*task, cmd_task);

        // create loss
        const auto loss = get_losses().get(cmd_loss);

        // create enhancer
        const auto enhancer = get_enhancers().get(cmd_enhancer, cmd_enhancer_params);

        // create model
        const auto model = get_models().get(cmd_model, cmd_model_params);
        model->configure(*task);
        model->random();
        model->describe();

        if (*model != *task)
        {
                log_error() << "mis-matching model and task!";
                return EXIT_FAILURE;
        }

        // create trainer
        const auto trainer = get_trainers().get(cmd_trainer, cmd_trainer_params);

        // train model
        accumulator_t acc(*model, *loss);
        acc.threads(cmd_threads);

        trainer_result_t result;
        measure_critical_and_log([&] ()
                {
                        result = trainer->train(*enhancer, *task, cmd_task_fold, acc);
                        return result.valid();
                },
                "train model");

        if (result.valid())
        {
                model->params(result.optimum_params());
        }

        // save the model & its optimization history
        measure_critical_and_log(
                [&] () { return model->save(cmd_model_file); },
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
