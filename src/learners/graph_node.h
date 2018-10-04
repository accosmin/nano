#pragma once

#include "arch.h"
#include "tensor.h"
#include "core/json.h"
#include "core/factory.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes.
        ///
        class graph_node_t;
        using layer_factory_t = factory_t<graph_node_t>;
        using rgraph_node_t = layer_factory_t::trobject;

        NANO_PUBLIC layer_factory_t& get_layers();

        ///
        /// \brief names for builtin computation nodes.
        ///
        inline const char* conv3d_node_name() { return "conv3d"; }
        inline const char* norm3d_node_name() { return "norm3d"; }
        inline const char* affine_node_name() { return "affine"; }
        inline const char* plus4d_node_name() { return "mix-plus"; }
        inline const char* tcat4d_node_name() { return "mix-tcat"; }

        ///
        /// \brief computation node.
        ///
        class NANO_PUBLIC graph_node_t : public json_configurable_t
        {
        public:

                ///
                /// \brief copy the current object
                ///
                virtual rgraph_node_t clone() const = 0;

                ///
                /// \brief configure to process tensors of the given size
                ///
                virtual bool resize(const tensor3d_dims_t& idims) = 0;

                ///
                /// \brief compute the output (given the input & the parameters)
                ///
                virtual void output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata) = 0;

                ///
                /// \brief compute the gradient wrt the inputs (given the output & the parameters)
                ///
                virtual void ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) = 0;

                ///
                /// \brief compute the (cumulated) gradient wrt the parameters (given the output & the input)
                ///
                virtual void gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata) = 0;

                ///
                /// \brief set parameters to random values
                ///
                virtual void random(vector_map_t pdata) const = 0;

                ///
                /// \brief returns the input/output/parameters dimensions
                ///
                virtual tensor_size_t psize() const = 0;
                virtual tensor3d_dim_t odims() const = 0;

                ///
                /// \brief returns the number of flops for the output/gparam/ginput operations
                ///
                virtual tensor_size_t flops_output() const = 0;
                virtual tensor_size_t flops_gparam() const = 0;
                virtual tensor_size_t flops_ginput() const = 0;
        };

        ///
        /// \brief convenience function to compute the size of the outputs of the given layer.
        ///
        inline auto osize(const graph_node_t& layer) { return nano::size(layer.odims()); }
        inline auto osize(const rgraph_node_t& layer) { return osize(*layer); }
}
