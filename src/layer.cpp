#include <mutex>
#include "layers/layer_affine.h"
#include "layers/layer_activation.h"
#include "layers/layer_convolution.h"

using namespace nano;

layer_factory_t& nano::get_layers()
{
        static layer_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<activation_layer_unit_t>("act-unit",      "activation: a(x) = x");
                m.add<activation_layer_sine_t>("act-sin",       "activation: a(x) = sin(x)");
                m.add<activation_layer_tanh_t>("act-tanh",      "activation: a(x) = tanh(x)");
                m.add<activation_layer_splus_t>("act-splus",    "activation: a(x) = log(1 + e^x)");
                m.add<activation_layer_snorm_t>("act-snorm",    "activation: a(x) = x / sqrt(1 + x^2)");
                m.add<activation_layer_sigm_t>("act-sigm",      "activation: a(x) = exp(x) / (1 + exp(x))");
                m.add<activation_layer_pwave_t>("act-pwave",    "activation: a(x) = x / (1 + x^2)");
                m.add<affine_layer_t>("affine",                 "transform:  L(x) = A * x + b");
                m.add<convolution_layer_t>("conv",              "transform:  L(x) = conv3D(x, kernel) + b");
        });

        return manager;
}
