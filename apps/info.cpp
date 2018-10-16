#include "loss.h"
#include "layer.h"
#include "model.h"
#include "solver.h"
#include "version.h"
#include "trainer.h"
#include "learner.h"
#include "core/table.h"
#include "core/cmdline.h"
#include "learners/activation.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tobject>
        void print(const string_t& name, const factory_t<tobject>& factory)
        {
                table_t table;
                table.header() << name << "description" << "configuration";
                table.delim();
                for (const auto& id : factory.ids())
                {
                        json_t json;
                        factory.get(id)->to_json(json);
                        table.append() << id << factory.description(id) << json.dump();
                }
                std::cout << table;
        }
}

int main(int argc, const char* argv[])
{
        // parse the command line
        cmdline_t cmdline("display the registered objects");
        cmdline.add("", "loss",                 "loss functions");
        cmdline.add("", "task",                 "tasks");
        cmdline.add("", "layer",                "layers to built models");
        cmdline.add("", "solver",               "numerical optimization algorithms");
        cmdline.add("", "trainer",              "training methods");
        cmdline.add("", "activation",           "activation functions (for ANN models)");
        cmdline.add("", "learner",              "machine learning models");
        cmdline.add("", "version",              "library version");
        cmdline.add("", "git-hash",             "git commit hash");
        cmdline.add("", "system",               "system: all available information");
        cmdline.add("", "sys-logical-cpus",     "system: number of logical cpus");
        cmdline.add("", "sys-physical-cpus",    "system: number of physical cpus");
        cmdline.add("", "sys-memsize",          "system: memory size in GB");

        cmdline.process(argc, argv);

        const auto has_loss = cmdline.has("loss");
        const auto has_task = cmdline.has("task");
        const auto has_layer = cmdline.has("layer");
        const auto has_solver = cmdline.has("solver");
        const auto has_trainer = cmdline.has("trainer");
        const auto has_activation = cmdline.has("activation");
        const auto has_learner = cmdline.has("learner");
        const auto has_system = cmdline.has("system");
        const auto has_sys_logical = cmdline.has("sys-logical-cpus");
        const auto has_sys_physical = cmdline.has("sys-physical-cpus");
        const auto has_sys_memsize = cmdline.has("sys-memsize");
        const auto has_version = cmdline.has("version");
        const auto has_git_hash = cmdline.has("git-hash");

        if (    !has_loss &&
                !has_task &&
                !has_layer &&
                !has_trainer &&
                !has_solver &&
                !has_activation &&
                !has_learner &&
                !has_system &&
                !has_sys_logical &&
                !has_sys_physical &&
                !has_sys_memsize &&
                !has_version &&
                !has_git_hash)
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
        if (has_trainer)
        {
                print("trainer", get_trainers());
        }
        if (has_activation)
        {
                print("activation", get_activations());
        }
        if (has_learner)
        {
                print("learner", get_learners());
        }
        if (has_solver)
        {
                print("solvers", get_solvers());
        }
        if (has_system || has_sys_physical)
        {
                std::cout << "physical CPUs..." << nano::physical_cpus() << std::endl;
        }
        if (has_system || has_sys_logical)
        {
                std::cout << "logical CPUs...." << nano::logical_cpus() << std::endl;
        }
        if (has_system || has_sys_memsize)
        {
                std::cout << "memsize........." << nano::memsize_gb() << "GB" << std::endl;
        }
        if (has_version)
        {
                std::cout << nano::major_version << "." << nano::minor_version << std::endl;
        }
        if (has_git_hash)
        {
                std::cout << nano::git_commit_hash << std::endl;
        }

        // OK
        return EXIT_SUCCESS;
}
