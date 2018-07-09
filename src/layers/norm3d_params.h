#pragma once

#include "tensor.h"
#include "core/cast.h"

namespace nano
{
        ///
        /// \brief normalization type.
        ///
        enum class norm_type
        {
                global,                 ///< globally using all feature planes
                plane,                  ///< per feature plane
        };

        template <>
        inline enum_map_t<norm_type> enum_string<norm_type>()
        {
                return
                {
                        { norm_type::global,    "global" },
                        { norm_type::plane,     "plane" }
                };
        }

        ///
        /// \brief parametrizes the normalization of 3D tensors.
        ///
        struct norm3d_params_t
        {
                norm3d_params_t(
                        const tensor_size_t xmaps = 0, const tensor_size_t xrows = 0, const tensor_size_t xcols = 0,
                        const norm_type type = norm_type::global) :
                        m_xmaps(xmaps), m_xrows(xrows), m_xcols(xcols),
                        m_ntype(type)
                {
                }

                norm3d_params_t(const tensor3d_dim_t& xdims, const norm_type type) :
                        norm3d_params_t(std::get<0>(xdims), std::get<1>(xdims), std::get<2>(xdims), type)
                {
                }

                auto xmaps() const { return m_xmaps; }
                auto xrows() const { return m_xrows; }
                auto xcols() const { return m_xcols; }
                auto xdims() const { return make_dims(xmaps(), xrows(), xcols()); }
                auto xdims(const tensor_size_t count) const { return make_dims(count, xmaps(), xrows(), xcols()); }
                auto xsize() const { return nano::size(xdims()); }

                auto psize() const { return 0; }

                auto ntype() const { return m_ntype; }

                auto flops_output() const { return 5 * xsize(); }
                auto flops_ginput() const { return 12 * xsize(); }
                auto flops_gparam() const { return 0; }

                auto make_xdata(const tensor_size_t count) const { return tensor4d_t(xdims(count)); }

                template <typename txdata>
                bool valid(const txdata&) const;

                bool valid() const { return xsize() > 0; }

                // attributes
                tensor_size_t   m_xmaps, m_xrows, m_xcols;
                norm_type       m_ntype;
        };

        template <typename txdata>
        inline bool norm3d_params_t::valid(const txdata& xdata) const
        {
                const auto count = xdata.template size<0>();
                return  valid() &&
                        xdata.template size<0>() == count &&
                        xdata.template size<1>() == xmaps() &&
                        xdata.template size<2>() == xrows() &&
                        xdata.template size<3>() == xcols();
        }

        inline bool operator==(const norm3d_params_t& params1, const norm3d_params_t& params2)
        {
                return  params1.m_xmaps == params2.m_xmaps &&
                        params1.m_xrows == params2.m_xrows &&
                        params1.m_xcols == params2.m_xcols &&
                        params1.m_ntype == params2.m_ntype;
        }
}
