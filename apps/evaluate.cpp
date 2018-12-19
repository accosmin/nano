#include "model.h"
#include "core/io.h"
#include "core/logger.h"
#include "core/cmdline.h"
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
        cmdline.add("", "model",      "path to the trained model (.model)");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_fold = cmdline.get<size_t>("fold");
        const auto cmd_loss = cmdline.get<string_t>("loss");
        const auto cmd_model = cmdline.get<string_t>("model");

        json_t json;
        string_t id;

        // load task
        critical(load_json(cmd_task, json, id),
                strcat("load task configuration from <", cmd_task, ">"));

        rtask_t task;
        critical(task = get_task(id),
                strcat("search task <", id, ">"));

        task->from_json(json);

        critical(task->load(),
                strcat("load task <", id, ">"));

        task->describe(id);

        // load loss
        critical(load_json(cmd_loss, json, id),
                strcat("load loss configuration from <", cmd_loss, ">"));

        rloss_t loss;
        critical(loss = get_loss(id),
                strcat("search loss <", id, ">"));

        // load model
        rmodel_t model;
        critical(model = model_t::load(cmd_model),
                strcat("load model from <", cmd_model, ">"));

        critical(*model == *task,
                strcat("checking model's compability with the task"));

        // test the model
        model_t::evaluate_t eval;
        critical(eval = model->evaluate(*task, fold_t{cmd_fold, protocol::test}, *loss),
                "evaluate model");

        // todo: add more stats (e.g. median, percentiles)
        log_info() << std::fixed << std::setprecision(3)
                << "error: " << eval.m_errors
                << ", loss: " << eval.m_values
                << ", " << eval.m_millis.count() << " ms/sample.";

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
