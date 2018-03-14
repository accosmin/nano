#include "io/io.h"
#include "checkpoint.h"
#include "accumulator.h"
#include "accumulator.h"
#include "text/cmdline.h"
#include <iomanip>

using namespace nano;

static bool load_json(const string_t& path, json_t& json)
{
        string_t config;
        return  load_string(path, config) &&
                (json == json_t::parse(config));
}

static bool load_json(const string_t& path, json_t& json, string_t& id)
{
        return  load_json(path, json) &&
                from_json(json, "type", id);
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("evaluate a model");
        cmdline.add("", "task",         join(get_tasks().ids()) + " (.json)");
        cmdline.add("", "fold",         "fold index to use for evaluation", "0");
        cmdline.add("", "loss",         join(get_losses().ids()) + " (.json)");
        cmdline.add("", "model",        "path to the trained model (.model)");
        cmdline.add("", "threads",      "number of threads to use", physical_cpus());

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_fold = cmdline.get<size_t>("fold");
        const auto cmd_model = cmdline.get<string_t>("model");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_threads = cmdline.get<size_t>("threads");

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

        // load model
        checkpoint.step(strcat("load model from <", cmd_model, ">"));

        model_t model;
        checkpoint.critical(model.load(cmd_model));

        model.random();
        model.describe();
        if (model != *task)
        {
                log_error() << "model not compatible with the task!";
                return EXIT_FAILURE;
        }

        // test model
        accumulator_t acc(model, *loss);
        acc.mode(accumulator_t::type::value);
        acc.threads(cmd_threads);

        checkpoint.step("evaluate model");
        acc.update(*task, fold_t{cmd_fold, protocol::test});
        checkpoint.measure();

        log_info() << std::fixed << std::setprecision(3)
                << "test=" << acc.vstats().avg() << "|" << acc.estats().avg() << "+/-" << acc.estats().var() << ".";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
