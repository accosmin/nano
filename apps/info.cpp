#include "text/cmdline.h"
#include "text/table.h"
#include "cortex/cortex.h"
#include <iostream>

namespace
{
        template
        <
                typename tobject
        >
        void print(const nano::string_t& name, const nano::manager_t<tobject>& manager)
        {
                using namespace nano;

                const strings_t ids = manager.ids();
                const strings_t descriptions = manager.descriptions();

                nano::table_t table(name);
                table.header() << "description";

                for (size_t i = 0; i < ids.size(); ++ i)
                {
                        table.append(ids[i]) << descriptions[i];
                }
                table.print(std::cout);
                std::cout << std::endl;
        }
}

int main(int argc, const char* argv[])
{
        // parse the command line
        nano::cmdline_t cmdline("display the registered objects");
        cmdline.add("", "loss",         "loss functions");
        cmdline.add("", "task",         "tasks");
        cmdline.add("", "layer",        "layer types to built models");
        cmdline.add("", "model",        "model types");
        cmdline.add("", "trainer",      "training methods");
        cmdline.add("", "criterion",    "training criteria");

        cmdline.process(argc, argv);

        const bool has_loss = cmdline.has("loss");
        const bool has_task = cmdline.has("task");
        const bool has_layer = cmdline.has("layer");
        const bool has_model = cmdline.has("model");
        const bool has_trainer = cmdline.has("trainer");
        const bool has_criterion = cmdline.has("criterion");

        if (    !has_loss &&
                !has_task &&
                !has_layer &&
                !has_model &&
                !has_trainer &&
                !has_criterion)
        {
                cmdline.usage();
                return EXIT_FAILURE;
        }

        // check arguments and options
        if (has_loss)
        {
                print("loss", nano::get_losses());
        }
        if (has_task)
        {
                print("task", nano::get_tasks());
        }
        if (has_layer)
        {
                print("layer", nano::get_layers());
        }
        if (has_model)
        {
                print("model", nano::get_models());
        }
        if (has_trainer)
        {
                print("trainer", nano::get_trainers());
        }
        if (has_criterion)
        {
                print("criterion", nano::get_criteria());
        }

        // OK
        return EXIT_SUCCESS;
}
