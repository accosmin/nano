#include "loss.h"
#include "task.h"
#include "model.h"
#include "io/io.h"
#include "trainer.h"
#include "checkpoint.h"
#include "text/table.h"
#include "accumulator.h"
#include "text/cmdline.h"
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
        cmdline_t cmdline("train a model");
        cmdline.add("", "task",         join(get_tasks().ids()) + " (.json)");
        cmdline.add("", "model",        "model configuration (.json)");
        cmdline.add("", "trainer",      join(get_trainers().ids()) + " (.json)");
        cmdline.add("", "loss",         join(get_losses().ids()) + " (.json)");
        cmdline.add("", "basepath",     "basepath where to save results (e.g. model, logs, history)");
        cmdline.add("", "threads",      "number of threads to use", physical_cpus());
        cmdline.add("", "trials",       "number of trials/folds", 10);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_trainer = cmdline.get<string_t>("trainer");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_basepath = cmdline.get<string_t>("basepath");
        const auto cmd_threads = cmdline.get<size_t>("threads");
        const auto cmd_trials = cmdline.get<size_t>("trials");

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

        describe(*task, id);

        // load loss
        checkpoint.step(strcat("load loss configuration from <", cmd_loss, ">"));
        checkpoint.critical(load_json(cmd_loss, json, id));

        rloss_t loss;
        checkpoint.step(strcat("search loss <", id, ">"));
        checkpoint.critical((loss = get_losses().get(id)) != nullptr);

        // load trainer
        checkpoint.step(strcat("load trainer configuration from <", cmd_trainer, ">"));
        checkpoint.critical(load_json(cmd_trainer, json, id));

        rtrainer_t trainer;
        checkpoint.step(strcat("search trainer <", id, ">"));
        checkpoint.critical((trainer = get_trainers().get(id)) != nullptr);

        trainer->from_json(json);

        // load model
        checkpoint.step(strcat("load model configuration from <", cmd_model, ">"));
        checkpoint.critical(load_json(cmd_model, json));

        model_t model;
        checkpoint.step("configure model");
        checkpoint.measure(model.from_json(json) && model.resize(task->idims(), task->odims()));

        model.describe();
        if (model != *task)
        {
                log_error() << "model not compatible with the task!";
                return EXIT_FAILURE;
        }

        // setup accumulator
        accumulator_t acc(model, *loss);
        acc.threads(cmd_threads);

        table_t table;
        table.header()
                << "trial" << "epoch"
                << "train_loss" << "train_error"
                << "valid_loss" << "valid_error"
                << "test_loss" << "test_error"
                << "xnorm" << "gnorm"
                << "seconds" << "speed";
        table.delim();

        // train & save the model using multiple trials
        for (size_t trial = 0; trial < cmd_trials; ++ trial)
        {
                acc.random();
                trainer_result_t result;
                checkpoint.step("train model");
                checkpoint.measure((result = trainer->train(*task, trial % task->fsize(), acc)) == true);

                model.params(result.optimum_params());

                const auto& state = result.optimum_state();
                table.append()
                        << precision(0) << (trial + 1) << state.m_epoch
                        << precision(3) << state.m_train.m_value << state.m_train.m_error
                        << precision(3) << state.m_valid.m_value << state.m_valid.m_error
                        << precision(3) << state.m_test.m_value << state.m_test.m_error
                        << precision(3) << state.m_xnorm << state.m_gnorm
                        << precision(0) << idiv(state.m_milis.count(), 1000)
                        << precision(6) << result.convergence_speed();

                checkpoint.step("save model");
                checkpoint.critical(
                        model.save(strcat(cmd_basepath, "_trial", trial + 1, ".model")) &&
                        result.save(strcat(cmd_basepath, "_trial", trial + 1, ".csv")));
        }

        checkpoint.step("save stats");
        checkpoint.critical(table.save(strcat(cmd_basepath, ".csv")));

        std::cout << table;

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
