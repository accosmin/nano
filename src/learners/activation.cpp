#include <mutex>
#include "activation.h"

using namespace nano;

activation_factory_t& nano::get_activations()
{
        static activation_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<activation_unit_t>("act-unit",        "activation: a(x) = x");
                manager.add<activation_sine_t>("act-sin",         "activation: a(x) = sin(x)");
                manager.add<activation_tanh_t>("act-tanh",        "activation: a(x) = tanh(x)");
                manager.add<activation_splus_t>("act-splus",      "activation: a(x) = log(1 + e^x)");
                manager.add<activation_snorm_t>("act-snorm",      "activation: a(x) = x / sqrt(1 + x^2)");
                manager.add<activation_ssign_t>("act-ssign",      "activation: a(x) = x / (1 + |x|)");
                manager.add<activation_sigm_t>("act-sigm",        "activation: a(x) = exp(x) / (1 + exp(x))");
                manager.add<activation_pwave_t>("act-pwave",      "activation: a(x) = x / (1 + x^2)");
        });

        return manager;
}
