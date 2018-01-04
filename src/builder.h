#pragma once

#include "model.h"
#include "text/algorithm.h"
#include "layers/norm3d_params.h"

namespace nano
{
        ///
        /// \brief check if a computation node is an activation node.
        ///
        inline bool is_activation_node(const string_t& node_id)
        {
                return nano::starts_with(node_id, "act-");
        }

        ///
        /// \brief configure computation nodes.
        ///
        inline void config_empty_node(json_writer_t& writer)
        {
                writer.object();
        }

        inline void config_norm3d_node(json_writer_t& writer, const norm_type type)
        {
                writer.object("type", type);
        }

        inline void config_conv3d_node(json_writer_t& writer,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn = 1, const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                writer.object("omaps", omaps, "krows", krows, "kcols", kcols, "kconn", kconn, "kdrow", kdrow, "kdcol", kdcol);
        }

        inline void config_affine_node(json_writer_t& writer,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols)
        {
                writer.object("omaps", omaps, "orows", orows, "ocols", ocols);
        }

        ///
        /// \brief serialize computation nodes.
        ///
        template <typename tname, typename ttype, typename tconfigurer, typename... targs>
        json_writer_t& add_node(json_writer_t& writer, const tname& name, const ttype& type,
                const tconfigurer& configurer, const targs&... args)
        {
                writer.new_object()
                      .pairs("name", name, "type", type).next()
                      .name("config");
                      configurer(writer, args...);
                return writer.end_object();
        }

        template <typename tname>
        json_writer_t& add_norm3d_node(json_writer_t& writer, const tname& name, const norm_type type)
        {
                return add_node(writer, name, norm3d_node_name(),
                        config_norm3d_node, type);
        }

        template <typename tname>
        json_writer_t& add_conv3d_node(json_writer_t& writer, const tname& name,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn = 1, const tensor_size_t kdrow = 1, const tensor_size_t kdcol = 1)
        {
                return add_node(writer, name, conv3d_node_name(),
                        config_conv3d_node, omaps, krows, kcols, kconn, kdrow, kdcol);
        }

        template <typename tname>
        json_writer_t& add_affine_node(json_writer_t& writer, const tname& name,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols)
        {
                return add_node(writer, name, affine_node_name(),
                        config_affine_node, omaps, orows, ocols);
        }

        template <typename tname, typename ttype>
        json_writer_t& add_activation_node(json_writer_t& writer, const tname& name, const ttype& type)
        {
                assert(is_activation_node(type));
                return add_node(writer, name, type,
                        config_empty_node);
        }

        ///
        /// \brief add a computation node to the model.
        ///
        template <typename top, typename... targs>
        bool add_node(model_t& model, const string_t& name, const string_t& type, const top& op, targs&&... args)
        {
                if (name.empty())
                {
                        return true;
                }
                else
                {
                        json_writer_t writer;
                        op(writer, std::forward<targs>(args)...);
                        return model.add(name, type, writer.str());
                }
        }

        ///
        /// \brief connect computation nodes in a model.
        ///     node1 -> node2 -> ... -> nodeN
        ///
        inline bool connect_nodes(model_t& model, const string_t& node1, const string_t& node2)
        {
                assert(!node2.empty());
                return node1.empty() || model.connect(node1, node2);
        }

        template <typename... tnames>
        bool connect_nodes(model_t& model, const string_t& node1, const string_t& node2, const tnames&... nodes)
        {
                return  connect_nodes(model, node1, node2) &&
                        connect_nodes(model, node2, nodes...);
        }

        ///
        /// \brief adds an affine module to a computation graph:
        ///     - a fully connected affine node followed by
        ///     - a non-linearity node (if given).
        ///
        inline bool add_affine_module(model_t& model,
                const string_t& affine_name, const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation_name, const string_t& activation_type,
                const string_t& previous_name)
        {
                assert(activation_name.empty() || is_activation_node(activation_type));

                return  add_node(model, affine_name, affine_node_name(), config_affine_node, omaps, orows, ocols) &&
                        add_node(model, activation_name, activation_type, config_empty_node) &&
                        connect_nodes(model, previous_name, affine_name, activation_name);
        }

        ///
        /// \brief adds an affine module to a computation graph:
        ///     - a fully connected affine node followed by
        ///     - a non-linearity node (if given).
        ///
        inline bool add_conv3d_module(model_t& model,
                const string_t& conv3d_name,
                const tensor_size_t omaps, const tensor_size_t krows, const tensor_size_t kcols,
                const tensor_size_t kconn, const tensor_size_t kdrow, const tensor_size_t kdcol,
                const string_t& activation_name, const string_t& activation_type,
                const string_t& previous_name)
        {
                assert(activation_name.empty() || is_activation_node(activation_type));

                return  add_node(model, conv3d_name, conv3d_node_name(), config_conv3d_node, omaps, krows, kcols, kconn, kdrow, kdcol) &&
                        add_node(model, activation_name, activation_type, config_empty_node) &&
                        connect_nodes(model, previous_name, conv3d_name, activation_name);
        }

        ///
        /// \brief adds a mixing module to a computation graph:
        ///     - a node that combines two inputs (e.g. by summing or by concatenation)
        ///
        inline bool add_mixing_module(model_t& model,
                const string_t& mix_name, const string_t& mix_type,
                const string_t& input1_name, const string_t& input2_name)
        {
                return  add_node(model, mix_name, mix_type, config_empty_node) &&
                        model.connect(input1_name, mix_name) &&
                        model.connect(input2_name, mix_name);
        }

        ///
        /// \brief adds an output module to a computation graph:
        ///     - a fully connected affine node.
        ///
        inline bool add_output_module(model_t& model,
                const string_t& output_name, const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& previous_name)
        {
                return  add_node(model, output_name, affine_node_name(), config_affine_node, omaps, orows, ocols) &&
                        connect_nodes(model, previous_name, output_name);
        }

        ///
        /// \brief configuration for an affine node: omaps, orows, ocols
        ///
        using affine_node_config_t = std::array<tensor_size_t, 3>;
        using affine_node_configs_t = std::vector<affine_node_config_t>;

        ///
        /// \brief configuration for a 3D convolution node: omaps, krows, kcols, kconn, kdrow, kdcol
        ///
        using conv3d_node_config_t = std::array<tensor_size_t, 6>;
        using conv3d_node_configs_t = std::vector<conv3d_node_config_t>;

        ///
        /// \brief create a convolution neural network (CNN) consisting of:
        ///     (optional) the convolution nodes: [convX -> actX]+
        ///     (optional) followed by the affine (fully connected) nodes: [affY -> actY]+
        ///     followed by the output (affine) node: -> out
        ///
        /// NB: a linear model is obtained if the convolution and the affine layers are empty
        /// NB: a multi-layer perceptron (MLP) model is obtained if the convolution layers are empty
        ///
        NANO_PUBLIC bool make_cnn(model_t& model,
                const conv3d_node_configs_t& conv3d_param,
                const affine_node_configs_t& affine_param,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation_type);

        ///
        /// \brief create a multi-layer perceptron (MLP) model: a CNN without convolution nodes.
        ///
        NANO_PUBLIC bool make_mlp(model_t& model,
                const affine_node_configs_t& affine_param,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation_type);

        ///
        /// \brief create a linear model: a CNN without convolution and affine nodes.
        ///
        NANO_PUBLIC bool make_linear(model_t& model,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation_type);

        ///
        /// \brief create a residual MLP (multi-layer perceptron) network given
        ///
        /// structure:
        ///     aff0 -> act0 -> aff1 ->  act1 -> aff2 ->  act2 -> aff3 ->  act3 -> ... ->  actN   -> out
        ///                             +act0            +act1            +act2           +actN-1
        ///
        NANO_PUBLIC bool make_residual_mlp(model_t& model,
                const affine_node_configs_t& affine_param,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation_type,
                const string_t& mixing_type = mix_plus4d_node_name());
}
