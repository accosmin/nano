#include "ncv.h"
#include "core/logger.h"

template
<
        typename tobject
>
void print(const ncv::string_t& name, const ncv::manager_t<tobject>& manager)
{
        const ncv::strings_t ids = manager.ids();
        const ncv::strings_t descriptions = manager.descriptions();

        std::cout << ncv::string_t(120, '-') << std::endl;
        std::cout << ncv::text::resize(name + " id", 16, ncv::align::left)
                  << ncv::text::resize(name + " description", 64, ncv::align::left) << std::endl;

        std::cout << ncv::string_t(120, '=') << std::endl;
        for (ncv::size_t i = 0; i < ids.size(); i ++)
        {
                std::cout << ncv::text::resize(ids[i], 16) << descriptions[i] << std::endl;
        }
        std::cout << std::endl;
}

int main(int argc, char *argv[])
{
        ncv::init();

        print("loss",           ncv::loss_manager_t::instance());
        print("activation",     ncv::activation_manager_t::instance());
        print("task",           ncv::task_manager_t::instance());
        print("model",          ncv::model_manager_t::instance());
        print("trainer",        ncv::trainer_manager_t::instance());

        // OK
        ncv::log_info() << ncv::done;
        return EXIT_SUCCESS;
}
