#include "loss.h"
#include "task.h"
#include "learner.h"
#include "core/io.h"
#include "core/table.h"
#include "core/cmdline.h"
#include "core/checkpoint.h"
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
                from_json(json, "id", id);
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("train a model");
        cmdline.add("", "task",         "task configuration (.json)");
        cmdline.add("", "loss",         "loss configuration (.json)");
        cmdline.add("", "learner",      "learner configuration (.json)");
        cmdline.add("", "basepath",     "basepath where to save results (e.g. model, logs, history)");
        cmdline.add("", "trials",       "number of trials/folds", 10);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_learner = cmdline.get<string_t>("learner");
        const auto cmd_basepath = cmdline.get<string_t>("basepath");
        const auto cmd_trials = cmdline.get<size_t>("trials");

        checkpoint_t checkpoint;
        json_t json;
        string_t task_id, loss_id, learner_id;

        // load task
        checkpoint.step(strcat("load task configuration from <", cmd_task, ">"));
        checkpoint.critical(load_json(cmd_task, json, task_id));

        rtask_t task;
        checkpoint.step(strcat("search task <", task_id, ">"));
        checkpoint.critical((task = get_tasks().get(task_id)) != nullptr);

        task->from_json(json);
        checkpoint.step(strcat("load task <", task_id, ">"));
        checkpoint.measure(task->load());

        task->describe(task_id);

        // load loss
        checkpoint.step(strcat("load loss configuration from <", cmd_loss, ">"));
        checkpoint.critical(load_json(cmd_loss, json, loss_id));

        rloss_t loss;
        checkpoint.step(strcat("search loss <", loss_id, ">"));
        checkpoint.critical((loss = get_losses().get(loss_id)) != nullptr);

        // load learner
        checkpoint.step(strcat("load learner configuration from <", cmd_learner, ">"));
        checkpoint.critical(load_json(cmd_learner, json, learner_id));

        rlearner_t learner;
        checkpoint.step(strcat("search learner <", learner_id, ">"));
        checkpoint.critical((learner = get_learners().get(learner_id)) != nullptr);

        learner->from_json(json);

        //
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
                trainer_result_t result;
                checkpoint.step("train");
                checkpoint.measure((result = learner->train(*task, trial % task->fsize(), *loss)) == true);

                const auto& state = result.optimum_state();
                table.append()
                        << precision(0) << (trial + 1) << state.m_epoch
                        << precision(3) << state.m_train.m_value << state.m_train.m_error
                        << precision(3) << state.m_valid.m_value << state.m_valid.m_error
                        << precision(3) << state.m_test.m_value << state.m_test.m_error
                        << precision(3) << state.m_xnorm << state.m_gnorm
                        << precision(0) << idiv(state.m_milis.count(), 1000)
                        << precision(6) << result.convergence_speed();

                const auto path_learner = strcat(cmd_basepath, "_trial", trial + 1, ".model");
                const auto path_training = strcat(cmd_basepath, "_trial", trial + 1, ".csv");

                checkpoint.step("save model");
                checkpoint.critical(learner_t::save(path_learner, learner_id, *learner));
                checkpoint.critical(result.save(path_training));
        }

        checkpoint.step("save stats");
        checkpoint.critical(table.save(strcat(cmd_basepath, ".csv")));

        std::cout << table;

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
