#pragma once

#include "task.h"
#include "layer.h"

namespace nano
{
        class model_t;
        using rmodel_t = std::unique_ptr<model_t>;

        ///
        /// \brief computation directed graph.
        ///
        class NANO_PUBLIC model_t
        {
        public:

                ///
                /// \brief constructors & asignment operators
                ///
                model_t() = default;
                model_t(const model_t&) = default;
                model_t(model_t&&) = default;
                model_t& operator=(model_t&&) = default;
                model_t& operator=(const model_t&) = delete;

                ///
                /// \brief copy the current object
                ///
                rmodel_t clone() const;

                ///
                /// \brief remove all computation nodes
                ///
                void clear();

                ///
                /// \brief add a new computation name given by its name, type and configuration
                ///
                bool add(const string_t& name, const string_t& type, json_reader_t&);
                bool add(const string_t& name, const string_t& type, const string_t& json);

                ///
                /// \brief connect two computation nodes such that the first one is an input of the second one
                ///
                bool connect(const string_t& name1, const string_t& name2);

                ///
                /// \brief configure the computation graph using JSON
                ///
                bool config(json_reader_t&);
                bool config(const string_t& json);

                ///
                /// \brief serialize the computation graph to JSON
                ///
                void config(json_writer_t&) const;

                ///
                /// \brief configuration done
                ///
                bool done();

                ///
                /// \brief resize to process the given input/output size
                ///
                bool resize(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims);

                ///
                /// \brief serialize model to disk
                ///
                bool save(const string_t& path) const;
                bool load(const string_t& path);

                ///
                /// \brief serialize parameters to memory
                ///
                vector_t params() const;
                void params(const vector_t&);

                ///
                /// \brief set parameters to random values
                ///
                void random();

                ///
                /// \brief compute the model's output given its input
                ///
                const tensor4d_t& output(const tensor4d_t& idata);

                ///
                /// \brief compute the model's gradient wrt parameters given its output
                ///
                const tensor1d_t& gparam(const tensor4d_t& odata);

                ///
                /// \brief compute the model's gradient wrt inputs given its output
                ///
                const tensor4d_t& ginput(const tensor4d_t& odata);

                ///
                /// \brief retrieve timing information for all components
                ///
                probes_t probes() const;

                ///
                /// \brief print a short description of the model
                ///
                void describe() const;

                ///
                /// \brief returns the input/output dimensions
                ///
                tensor3d_dims_t idims() const;
                tensor3d_dims_t odims() const;

                ///
                /// \brief number of parameters (to optimize)
                ///
                tensor_size_t psize() const;

        private:

                bool config_nodes(json_reader_t&);
                bool config_model(json_reader_t&);

                ///
                /// \brief computation node.
                ///
                struct cnode_t
                {
                        cnode_t() = default;
                        cnode_t(const cnode_t&);
                        cnode_t(cnode_t&&) = default;
                        cnode_t& operator=(const cnode_t&);
                        cnode_t& operator=(cnode_t&&) = default;
                        cnode_t(const string_t& name, const string_t& type, rlayer_t&&);

                        string_t        m_name;
                        string_t        m_type;
                        rlayer_t        m_node;         ///< the computation node
                        indices_t       m_inodes;       ///< input computation nodes
                        indices_t       m_onodes;       ///< output computation nodes
                };

                const cnode_t& inode() const;
                const cnode_t& onode() const;
                size_t find_node(const string_t& name) const;
                bool resize_node(const size_t index, const tensor3d_dims_t& idims) const;
                void output_node(const size_t index, const tensor4d_t& idata) const;
                void ginput_node(const size_t index, const tensor4d_t& odata) const;
                void gparam_node(const size_t index, const tensor4d_t& odata) const;

                using cnodes_t = std::vector<cnode_t>;

                // attributes
                cnodes_t        m_nodes;
                size_t          m_inode{0};             ///< input (source) node
                size_t          m_onode{0};             ///< output (sink) node
                tensor1d_t      m_gdata;                ///< gradient wrt parameters
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };

        ///
        /// \brief check if the given model is compatible with the given task.
        ///
        inline bool operator==(const model_t& model, const task_t& task)
        {
                return  model.idims() == task.idims() &&
                        model.odims() == task.odims();
        }

        inline bool operator!=(const model_t& model, const task_t& task)
        {
                return !(model == task);
        }

        ///
        /// \brief resize the model to process samples from the given task
        ///
        inline bool resize(model_t& model, const task_t& task)
        {
                return model.resize(task.idims(), task.odims());
        }

        ///
        /// \brief convenience function to compute the size of the inputs / outputs of the given model.
        ///
        inline auto isize(const model_t& model) { return nano::size(model.idims()); }
        inline auto osize(const model_t& model) { return nano::size(model.odims()); }

        inline auto isize(const model_t& model, const tensor_size_t count) { return count * isize(model); }
        inline auto osize(const model_t& model, const tensor_size_t count) { return count * osize(model); }

        inline auto idims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.idims()); }
        inline auto odims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.odims()); }
}
