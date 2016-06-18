#include "text/cmdline.h"
#include "text/table.h"
#include "cortex/cortex.h"
#include "optim/batch/types.h"
#include "optim/stoch/types.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tobject>
        void print(const string_t& name, const manager_t<tobject>& manager)
        {
                const strings_t ids = manager.ids();
                const strings_t descriptions = manager.descriptions();

                table_t table(name);
                table.header() << "description";

                for (size_t i = 0; i < ids.size(); ++ i)
                {
                        table.append(ids[i]) << descriptions[i];
                }
                table.print(std::cout);
                std::cout << std::endl;
        }

        template <typename tenum>
        void print(const string_t& name)
        {
                const strings_t ids = enum_strings<tenum>();

                table_t table(name);

                for (size_t i = 0; i < ids.size(); ++ i)
                {
                        table.append(ids[i]);
                }
                table.print(std::cout);
                std::cout << std::endl;
        }
}

int main(int argc, const char* argv[])
{
        // parse the command line
        cmdline_t cmdline("display the registered objects");
        cmdline.add("", "loss",         "loss functions");
        cmdline.add("", "task",         "tasks");
        cmdline.add("", "layer",        "layer types to built models");
        cmdline.add("", "model",        "model types");
        cmdline.add("", "trainer",      "training methods");
        cmdline.add("", "criterion",    "training criteria");
        cmdline.add("", "batch",        "batch optimization algorithms");
        cmdline.add("", "stoch",        "stochastic optimization algorithms");

        cmdline.process(argc, argv);

        const bool has_loss = cmdline.has("loss");
        const bool has_task = cmdline.has("task");
        const bool has_layer = cmdline.has("layer");
        const bool has_model = cmdline.has("model");
        const bool has_trainer = cmdline.has("trainer");
        const bool has_criterion = cmdline.has("criterion");
        const bool has_batch = cmdline.has("batch");
        const bool has_stoch = cmdline.has("stoch");

        if (    !has_loss &&
                !has_task &&
                !has_layer &&
                !has_model &&
                !has_trainer &&
                !has_criterion &&
                !has_batch &&
                !has_stoch)
        {
                cmdline.usage();
                return EXIT_FAILURE;
        }

        // check arguments and options
        if (has_loss)
        {
                print("loss", get_losses());
        }
        if (has_task)
        {
                print("task", get_tasks());
        }
        if (has_layer)
        {
                print("layer", get_layers());
        }
        if (has_model)
        {
                print("model", get_models());
        }
        if (has_trainer)
        {
                print("trainer", get_trainers());
        }
        if (has_criterion)
        {
                print("criterion", get_criteria());
        }
        if (has_batch)
        {
                print<batch_optimizer>("batch optimizer");
        }
        if (has_stoch)
        {
                print<stoch_optimizer>("stochastic optimizer");
        }
        // OK
        return EXIT_SUCCESS;
}
