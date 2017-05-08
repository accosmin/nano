#include "nano.h"
#include "version.h"
#include "text/table.h"
#include "text/cmdline.h"
#include <iostream>

using namespace nano;

namespace
{
        template <typename tobject>
        void print(const string_t& name, const manager_t<tobject>& manager)
        {
                const auto ids = manager.ids();
                const auto descriptions = manager.descriptions();
                const auto configurations = manager.configs();

                table_t table;
                table.header() << name << "description" << "configuration";

                for (size_t i = 0; i < ids.size(); ++ i)
                {
                        table.append() << ids[i] << descriptions[i] << configurations[i];
                }
                std::cout << table << std::endl;
        }
}

int main(int argc, const char* argv[])
{
        // parse the command line
        cmdline_t cmdline("display the registered objects");
        cmdline.add("", "loss",                 "loss functions");
        cmdline.add("", "task",                 "tasks");
        cmdline.add("", "layer",                "layer types to built models");
        cmdline.add("", "model",                "model types");
        cmdline.add("", "sampler",              "training sampling methods");
        cmdline.add("", "trainer",              "training methods");
        cmdline.add("", "batch",                "batch optimization algorithms");
        cmdline.add("", "stoch",                "stochastic optimization algorithms");
        cmdline.add("", "version",              "library version");
        cmdline.add("", "git-hash",             "git commit hash");
        cmdline.add("", "system",               "system: all available information");
        cmdline.add("", "sys-logical-cpus",     "system: number of logical cpus");
        cmdline.add("", "sys-physical-cpus",    "system: number of physical cpus");
        cmdline.add("", "sys-memsize",          "system: memory size in GB");

        cmdline.process(argc, argv);

        const bool has_loss = cmdline.has("loss");
        const bool has_task = cmdline.has("task");
        const bool has_layer = cmdline.has("layer");
        const bool has_model = cmdline.has("model");
        const bool has_sampler = cmdline.has("sampler");
        const bool has_trainer = cmdline.has("trainer");
        const bool has_batch = cmdline.has("batch");
        const bool has_stoch = cmdline.has("stoch");
        const bool has_system = cmdline.has("system");
        const bool has_sys_logical = cmdline.has("sys-logical-cpus");
        const bool has_sys_physical = cmdline.has("sys-physical-cpus");
        const bool has_sys_memsize = cmdline.has("sys-memsize");
        const bool has_version = cmdline.has("version");
        const bool has_git_hash = cmdline.has("git-hash");

        if (    !has_loss &&
                !has_task &&
                !has_layer &&
                !has_model &&
                !has_sampler &&
                !has_trainer &&
                !has_batch &&
                !has_stoch &&
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
        if (has_model)
        {
                print("model", get_models());
        }
        if (has_sampler)
        {
                print("sampler", get_samplers());
        }
        if (has_trainer)
        {
                print("trainer", get_trainers());
        }
        if (has_batch)
        {
                print("batch optimizers", get_batch_optimizers());
        }
        if (has_stoch)
        {
                print("stochastic optimizers", get_stoch_optimizers());
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
