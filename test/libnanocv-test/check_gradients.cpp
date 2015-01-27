#include "check_gradients.h"
#include "util/random.hpp"
#include "loss.h"
#include <set>

namespace test
{
        using namespace ncv;

        namespace
        {
                const strings_t conv_layer_ids { "", "conv" };
                const strings_t conv_masks { "25", "50", "100" };
                const strings_t pool_layer_ids { "", "pool-max", "pool-min", "pool-avg" };
                const strings_t full_layer_ids { "", "linear" };
                const strings_t actv_layer_ids { "", "act-unit", "act-tanh", "act-snorm", "act-splus" };

                const color_mode cmd_color = color_mode::luma;
                const size_t cmd_irows = 8;
                const size_t cmd_icols = 8;
                const size_t cmd_outputs = 4;
                const size_t cmd_max_layers = 2;

                string_t make_model_description(
                        size_t n_layers,
                        const string_t& actv_layer_id,
                        const string_t& pool_layer_id,
                        const string_t& conv_layer_id,
                        const string_t& conv_mask,
                        const string_t& full_layer_id)
                {
                        string_t desc;

                        // convolution part
                        for (size_t l = 0; l < n_layers && !conv_layer_id.empty(); l ++)
                        {
                                random_t<size_t> rgen(2, 3);

                                string_t params;
                                params += "dims=" + text::to_string(rgen());
                                params += (rgen() % 2 == 0) ? ",rows=2,cols=2," : ",rows=3,cols=3,";
                                params += "mask=" + conv_mask;

                                desc += conv_layer_id + ":" + params + ";";
                                if (l == 0)
                                {
                                        desc += pool_layer_id + ";";
                                }
                                desc += actv_layer_id + ";";
                        }

                        // fully-connected part
                        for (size_t l = 0; l < n_layers && !full_layer_id.empty(); l ++)
                        {
                                random_t<size_t> rgen(1, 5);

                                string_t params;
                                params += "dims=" + text::to_string(rgen());

                                desc += full_layer_id + ":" + params + ";";
                                desc += actv_layer_id + ";";
                        }

                        desc += "linear:dims=" + text::to_string(cmd_outputs) + ";";

                        return desc;
                }
        }

        std::vector<std::pair<ncv::string_t, ncv::string_t> > make_grad_configs(
                ncv::size_t& irows, ncv::size_t& icols, ncv::size_t& outputs, ncv::color_mode& color)
        {
                // evaluate the analytical gradient vs. the finite difference approximation for various:
                //      * convolution layers
                //      * convolution connection types
                //      * pooling layers
                //      * fully connected layers
                //      * activation layers
                std::set<string_t> descs;
                for (size_t n_layers = 0; n_layers <= cmd_max_layers; n_layers ++)
                {
                        for (const string_t& actv_layer_id : actv_layer_ids)
                        {
                                for (const string_t& pool_layer_id : pool_layer_ids)
                                {
                                        for (const string_t& conv_layer_id : conv_layer_ids)
                                        {
                                                for (const string_t& conv_mask : conv_masks)
                                                {
                                                        for (const string_t& full_layer_id : full_layer_ids)
                                                        {
                                                                const string_t desc = make_model_description(
                                                                        n_layers,
                                                                        actv_layer_id,
                                                                        pool_layer_id,
                                                                        conv_layer_id,
                                                                        conv_mask,
                                                                        full_layer_id);

                                                                descs.insert(desc);
                                                        }
                                                }
                                        }
                                }
                        }
                }

                const strings_t loss_ids = loss_manager_t::instance().ids();

                // create the <model description, loss id> configuration
                std::vector<std::pair<ncv::string_t, ncv::string_t> > result;
                for (const string_t& desc : descs)
                {
                        for (const string_t& loss_id : loss_ids)
                        {
                                result.emplace_back(desc, loss_id);
                        }
                }

                // OK
                irows = cmd_irows;
                icols = cmd_icols;
                outputs = cmd_outputs;
                color = cmd_color;
                return result;
        }
}
	
