#include "nanocv/nanocv.h"
#include "nanocv/tabulator.h"
#include <iostream>

template
<
        typename tobject
>
void print(const ncv::string_t& name, const ncv::manager_t<tobject>& manager)
{
        using namespace ncv;

        const strings_t ids = manager.ids();
        const strings_t descriptions = manager.descriptions();

        tabulator_t table(name);
        table.header() << "description";

        for (size_t i = 0; i < ids.size(); i ++)
        {
                table.append(ids[i]) << descriptions[i];
        }
        table.print(std::cout);
        std::cout << std::endl;
}

int main(int, char* [])
{
        ncv::init();

        print("loss",           ncv::get_losses());
        print("task",           ncv::get_tasks());
        print("layer",          ncv::get_layers());
        print("model",          ncv::get_models());
        print("trainer",        ncv::get_trainers());
        print("criterion",      ncv::get_criteria());

        // OK
        return EXIT_SUCCESS;
}
