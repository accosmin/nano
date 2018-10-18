#include "learner.h"
#include "core/io.h"
#include "core/cmdline.h"
#include "core/checkpoint.h"
#include <iomanip>

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
        cmdline_t cmdline("evaluate a model");
        cmdline.add("", "task",         "task configuration (.json)");
        cmdline.add("", "fold",         "fold index to use for evaluation", "0");
        cmdline.add("", "loss",         "loss configuration (.json)");
        cmdline.add("", "learner",      "path to the trained model (.model)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_fold = cmdline.get<size_t>("fold");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_learner = cmdline.get<string_t>("learner");

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

        // load learner
        checkpoint.step(strcat("load learner from <", cmd_learner, ">"));

        rlearner_t learner;
        checkpoint.critical((learner = learner_t::load(cmd_learner)) != nullptr);
        checkpoint.critical(*learner == *task);

        // test the learner
        checkpoint.step("evaluate learner");

        const auto fold = fold_t{cmd_fold, protocol::test};

        // todo: use the thread pool to speed-up computation
        stats_t stats_errors, stats_values;
        for (size_t i = 0, size = task->size(fold); i < size; ++ i)
        {
                const auto input = task->input(fold, i);
                const auto target = task->target(fold, i);
                const auto output = learner->output(input);

                stats_errors(loss->error(target, output));
                stats_values(loss->value(target, output));
        }

        checkpoint.measure();

        // todo: add more stats (e.g. median, percentiles)
        log_info() << std::fixed << std::setprecision(3)
                << "error: " << stats_errors
                << ", loss: " << stats_values << ".";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
