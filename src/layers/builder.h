#pragma once

#include "norm3d_params.h"
#include "text/algorithm.h"
#include "text/json_writer.h"

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
        /// \brief names for builtin computation nodes.
        ///
        inline const char* conv3d_node_name() { return "conv3d"; }
        inline const char* norm3d_node_name() { return "norm3d"; }
        inline const char* affine_node_name() { return "affine"; }
        inline const char* mix_plus4d_node_name() { return "mix-plus"; }
        inline const char* mix_tcat4d_node_name() { return "mix-tcat"; }

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
        template <typename tmodel, typename top, typename... targs>
        bool add_node(tmodel& model, const string_t& name, const string_t& type, const top& op, targs&&... args)
        {
                json_writer_t writer;
                op(writer, std::forward<targs>(args)...);
                return model.add(name, type, writer.str());
        }

        ///
        /// \brief create a MLP (multi-layer perceptron) network given
        /// \param affine_maps the number of outputs (aka feature maps) for each affine layer
        /// \param omaps the number of feature maps of the output layer
        /// \param orows the number of rows of the output layer
        /// \param ocols the number of columns of the output layer
        /// \param activation_type the type of the activation layer inserted after each affine layer
        ///
        /// structure:
        ///     affine0 -> activation0 -> affine1 -> activation1 -> ... -> activationN -> output
        ///
        template <typename tmodel, typename tmaps>
        void make_mlp(tmodel& model, const tmaps& affine_maps,
                const tensor_size_t omaps, const tensor_size_t orows, const tensor_size_t ocols,
                const string_t& activation_type)
        {
                assert(is_activation_node(activation_type));

                string_t last_affine_name;
                string_t last_activation_name;

                tensor_size_t n = 0;
                for (const tensor_size_t maps : affine_maps)
                {
                        const string_t affine_name = "affine" + to_string(n);
                        const string_t activation_name = "activation" + to_string(n);

                        const tensor_size_t rows = 1;
                        const tensor_size_t cols = 1;

                        add_node(model, affine_name, affine_node_name(), config_affine_node, maps, rows, cols);
                        add_node(model, activation_name, activation_type, config_empty_node);
                        model.connect(affine_name, activation_name);

                        if (n > 0)
                        {
                                model.connect(last_activation_name, affine_name);
                        }

                        ++ n;
                        last_affine_name = affine_name;
                        last_activation_name = activation_name;
                }

                add_node(model, "output", affine_node_name(), config_affine_node, omaps, orows, ocols);
                model.connect(last_activation_name, "output");
        }
}
