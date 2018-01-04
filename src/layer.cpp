#include <mutex>
#include "layers/layer_affine.h"
#include "layers/layer_norm3d.h"
#include "layers/layer_conv3d.h"
#include "layers/layer_plus4d.h"
#include "layers/layer_tcat4d.h"
#include "layers/layer_activation.h"

using namespace nano;

layer_factory_t& nano::get_layers()
{
        static layer_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<activation_layer_unit_t>("act-unit",        "activation: a(x) = x");
                manager.add<activation_layer_sine_t>("act-sin",         "activation: a(x) = sin(x)");
                manager.add<activation_layer_tanh_t>("act-tanh",        "activation: a(x) = tanh(x)");
                manager.add<activation_layer_splus_t>("act-splus",      "activation: a(x) = log(1 + e^x)");
                manager.add<activation_layer_snorm_t>("act-snorm",      "activation: a(x) = x / sqrt(1 + x^2)");
                manager.add<activation_layer_ssign_t>("act-ssign",      "activation: a(x) = x / (1 + |x|)");
                manager.add<activation_layer_sigm_t>("act-sigm",        "activation: a(x) = exp(x) / (1 + exp(x))");
                manager.add<activation_layer_pwave_t>("act-pwave",      "activation: a(x) = x / (1 + x^2)");
                manager.add<affine_layer_t>(affine_node_name(),         "transform:  L(x) = A * x + b");
                manager.add<conv3d_layer_t>(conv3d_node_name(),         "transform:  L(x) = conv3D(x, kernel) + b");
                manager.add<norm3d_layer_t>(norm3d_node_name(),         "transform: zero-mean & unit-variance");
                manager.add<plus4d_layer_t>(mix_plus4d_node_name(),     "combine: sum 4D inputs");
                manager.add<tcat4d_layer_t>(mix_tcat4d_node_name(),     "combine: concat 4D inputs across feature planes");
        });

        return manager;
}
