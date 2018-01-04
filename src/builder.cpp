#include "builder.h"

bool nano::make_cnn(model_t& model,
        const conv3d_node_configs_t& conv3d_param,
        const affine_node_configs_t& affine_param,
        const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
        const string_t& activation_type)
{
        assert(is_activation_node(activation_type));

        tensor_size_t depth = 0;
        string_t prev_activation_name;

        for (const auto& conv3d_node : conv3d_param)
        {
                const string_t conv3d_name = "conv3d" + to_string(depth);
                const string_t activation_name = "nonlin" + to_string(depth);

                const auto maps = std::get<0>(conv3d_node);
                const auto rows = std::get<1>(conv3d_node);
                const auto cols = std::get<2>(conv3d_node);
                const auto conn = std::get<3>(conv3d_node);
                const auto drow = std::get<4>(conv3d_node);
                const auto dcol = std::get<5>(conv3d_node);

                if (!add_conv3d_module(model,
                        conv3d_name, maps, rows, cols, conn, drow, dcol,
                        activation_name, activation_type,
                        prev_activation_name))
                {
                        return false;
                }

                ++ depth;
                prev_activation_name = activation_name;
        }

        for (const auto& affine_node : affine_param)
        {
                const string_t affine_name = "affine" + to_string(depth);
                const string_t activation_name = "nonlin" + to_string(depth);

                const auto maps = std::get<0>(affine_node);
                const auto rows = std::get<1>(affine_node);
                const auto cols = std::get<2>(affine_node);

                if (!add_affine_module(model,
                        affine_name, maps, rows, cols,
                        activation_name, activation_type,
                        prev_activation_name))
                {
                        return false;
                }

                ++ depth;
                prev_activation_name = activation_name;
        }

        return add_output_module(model, "output", omaps, orows, ocols, prev_activation_name);
}

bool nano::make_mlp(model_t& model,
        const affine_node_configs_t& affine_param,
        const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
        const string_t& activation_type)
{
        return make_cnn(model, {}, affine_param, omaps, orows, ocols, activation_type);
}

bool nano::make_linear(model_t& model,
        const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
        const string_t& activation_type)
{
        return make_cnn(model, {}, {}, omaps, orows, ocols, activation_type);
}

bool nano::make_residual_mlp(model_t& model,
        const affine_node_configs_t& affine_param,
        const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
        const string_t& activation_type, const string_t& mixing_type)
{
        assert(is_activation_node(activation_type));

        string_t prev_activation_name, prev_prev_activation_name, prev_mixing_name;

        bool ok = true;
        tensor_size_t depth = 0;
        for (const auto& affine_node : affine_param)
        {
                const string_t affine_name = "affine" + to_string(depth);
                const string_t activation_name = "nonlin" + to_string(depth);
                const string_t mixing_name = "mixing" + to_string(depth);

                const auto maps = std::get<0>(affine_node);
                const auto rows = std::get<1>(affine_node);
                const auto cols = std::get<2>(affine_node);

                const string_t* prev_name = &prev_activation_name;
                if (!prev_prev_activation_name.empty())
                {
                        // mix previous two affine modules
                        if (!(ok = add_mixing_module(model, mixing_name, mixing_type,
                                prev_activation_name,
                                prev_mixing_name.empty() ? prev_prev_activation_name : prev_mixing_name)))
                        {
                                break;
                        }

                        prev_name = &mixing_name;
                        prev_mixing_name = mixing_name;
                }

                if (!(ok = add_affine_module(
                        model, affine_name, maps, rows, cols,
                        activation_name, activation_type,
                        *prev_name)))
                {
                        break;
                }

                ++ depth;
                prev_prev_activation_name = prev_activation_name;
                prev_activation_name = activation_name;
        }

        return ok && add_output_module(model, "out", omaps, orows, ocols, prev_activation_name);
}

