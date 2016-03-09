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
        void print(const zob::string_t& name, const zob::manager_t<tobject>& manager)
        {
                using namespace zob;

                const strings_t ids = manager.ids();
                const strings_t descriptions = manager.descriptions();

                zob::table_t table(name);
                table.header() << "description";

                for (size_t i = 0; i < ids.size(); ++ i)
                {
                        table.append(ids[i]) << descriptions[i];
                }
                table.print(std::cout);
                std::cout << std::endl;
        }
}

int main(int argc, char* argv[])
{
        zob::init();

        // parse the command line
        zob::cmdline_t cmdline("display the registered objects");
        cmdline.add("", "loss",         "loss functions");
        cmdline.add("", "task",         "tasks");
        cmdline.add("", "layer",        "layer types to built models");
        cmdline.add("", "model",        "model types");
        cmdline.add("", "trainer",      "training methods");
        cmdline.add("", "criterion",    "training criteria");

        cmdline.process(argc, argv);

        // check arguments and options
        if (cmdline.has("loss"))
        {
                print("loss", zob::get_losses());
        }
        if (cmdline.has("task"))
        {
                print("task", zob::get_tasks());
        }
        if (cmdline.has("layer"))
        {
                print("layer", zob::get_layers());
        }
        if (cmdline.has("model"))
        {
                print("model", zob::get_models());
        }
        if (cmdline.has("trainer"))
        {
                print("trainer", zob::get_trainers());
        }
        if (cmdline.has("criterion"))
        {
                print("criterion", zob::get_criteria());
        }

        // OK
        return EXIT_SUCCESS;
}
