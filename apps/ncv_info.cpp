#include "libnanocv/nanocv.h"
#include "libnanocv/table.h"
#include <iostream>
#include <boost/program_options.hpp>

namespace
{
        template
        <
                typename tobject
        >
        void print(const ncv::string_t& name, const ncv::manager_t<tobject>& manager)
        {
                using namespace ncv;

                const strings_t ids = manager.ids();
                const strings_t descriptions = manager.descriptions();

                table_t table(name);
                table.header() << "description";

                for (size_t i = 0; i < ids.size(); i ++)
                {
                        table.append(ids[i]) << descriptions[i];
                }
                table.print(std::cout);
                std::cout << std::endl;
        }
}

int main(int argc, char* argv[])
{
        ncv::init();

        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h",         "display the registered objects");
        po_desc.add_options()("loss",           "loss functions");
        po_desc.add_options()("task",           "tasks");
        po_desc.add_options()("layer",          "layer types to built models");
        po_desc.add_options()("model",          "model types");
        po_desc.add_options()("trainer",        "training methods");
        po_desc.add_options()("criterion",      "training criteria");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (	po_vm.empty() ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        if (po_vm.count("loss"))
        {
                print("loss", ncv::get_losses());
        }
        if (po_vm.count("task"))
        {
                print("task", ncv::get_tasks());
        }
        if (po_vm.count("layer"))
        {
                print("layer", ncv::get_layers());
        }
        if (po_vm.count("model"))
        {
                print("model", ncv::get_models());
        }
        if (po_vm.count("trainer"))
        {
                print("trainer", ncv::get_trainers());
        }
        if (po_vm.count("criterion"))
        {
                print("criterion", ncv::get_criteria());
        }

        // OK
        return EXIT_SUCCESS;
}
