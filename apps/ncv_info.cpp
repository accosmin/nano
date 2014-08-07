#include "nanocv.h"

using namespace ncv;

template
<
        typename tobject
>
void print(const string_t& name, const manager_t<tobject>& manager)
{
        const strings_t ids = manager.ids();
        const strings_t descriptions = manager.descriptions();

        std::cout << string_t(120, '-') << std::endl;
        std::cout << text::resize(name + " id", 24, align::left)
                  << text::resize(name + " description", 64, align::left) << std::endl;

        std::cout << string_t(120, '=') << std::endl;
        for (size_t i = 0; i < ids.size(); i ++)
        {
                std::cout << text::resize(ids[i], 24) << descriptions[i] << std::endl;
        }
        std::cout << std::endl << std::endl;
}

int main(int argc, char *argv[])
{
        ncv::init();

        print("loss",           ncv::loss_manager_t::instance());
        print("task",           ncv::task_manager_t::instance());
        print("layer",          ncv::layer_manager_t::instance());
        print("model",          ncv::model_manager_t::instance());
        print("trainer",        ncv::trainer_manager_t::instance());
        print("criterion",      ncv::criterion_manager_t::instance());

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
