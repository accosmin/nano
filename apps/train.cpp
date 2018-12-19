#include "loss.h"
#include "task.h"
#include "model.h"
#include "core/io.h"
#include "core/table.h"
#include "core/logger.h"
#include "core/cmdline.h"
#include "core/numeric.h"
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
        cmdline.add("", "model",        "model configuration (.json)");
        cmdline.add("", "basepath",     "basepath where to save results (e.g. model, logs, history)");
        cmdline.add("", "trials",       "number of trials/folds", 10);

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_basepath = cmdline.get<string_t>("basepath");
        const auto cmd_trials = cmdline.get<size_t>("trials");

        json_t json;
        string_t task_id, loss_id, model_id;

        // load task
        critical(load_json(cmd_task, json, task_id),
                strcat("load task configuration from <", cmd_task, ">"));

        rtask_t task;
        critical(task = get_task(task_id),
                strcat("search task <", task_id, ">"));

        task->from_json(json);

        critical(task->load(),
                strcat("load task <", task_id, ">"));

        task->describe(task_id);

        // load loss
        critical(load_json(cmd_loss, json, loss_id),
                strcat("load loss configuration from <", cmd_loss, ">"));

        rloss_t loss;
        critical(loss = get_loss(loss_id),
                strcat("search loss <", loss_id, ">"));

        // load model
        critical(load_json(cmd_model, json, model_id),
                strcat("load model configuration from <", cmd_model, ">"));

        rmodel_t model;
        critical(model = get_model(model_id),
                strcat("search model <", model_id, ">"));

        model->from_json(json);

        //
        table_t table;
        table.header()
                << "trial" << "epoch"
                << "train_loss" << "train_error"
                << "valid_loss" << "valid_error"
                << "test_loss" << "test_error"
                << "seconds" << "speed";
        table.delim();

        // train & save the model using multiple trials
        for (size_t trial = 0; trial < cmd_trials; ++ trial)
        {
                training_t training;
                critical(training = model->train(*task, trial % task->fsize(), *loss),
                        "train");

                const auto& state = training.optimum();
                table.append()
                        << precision(0) << (trial + 1) << state.m_epoch
                        << precision(3) << state.m_train.m_value << state.m_train.m_error
                        << precision(3) << state.m_valid.m_value << state.m_valid.m_error
                        << precision(3) << state.m_test.m_value << state.m_test.m_error
                        << precision(0) << idiv(state.m_milis.count(), 1000)
                        << precision(6) << training.convergence_speed();

                const auto path_model = strcat(cmd_basepath, "_trial", trial + 1, ".model");
                const auto path_training = strcat(cmd_basepath, "_trial", trial + 1, ".csv");

                critical(model_t::save(path_model, model_id, *model), "save model");
                critical(training.save(path_training), "save training history");
        }

        critical(table.save(strcat(cmd_basepath, ".csv")), "save statistics");

        std::cout << table;

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
