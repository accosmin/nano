#pragma once

#include "norm_params.h"

namespace nano
{
        ///
        /// \brief normalizes 3D inputs to have zero mean and unit variance
        ///     either globally or per feature plane.
        ///
        /// parameters:
        ///     idata: 4D input tensor (count x imaps x irows x icols, with isize = imaps x irows x icols)
        ///     odata: 4D output tensor (count x omaps x orows x ocols, with osize = omaps x orows x ocols)
        ///
        /// operation:
        ///     odata = norm(idata)
        ///
        class norm4d_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit norm4d_t(const norm_params_t& params = norm_params_t()) :
                        m_params(params) {}

                ///
                /// \brief output
                ///
                template <typename tidata, typename todata>
                void output(const tidata& idata, todata&& odata) const;

                ///
                /// \brief gradient wrt inputs
                ///
                template <typename tidata, typename todata>
                void ginput(tidata&& idata, const todata& odata) const;

                ///
                /// \brief parameters
                ///
                const norm_params_t& params() const { return m_params; }

        private:

                template <typename tiarray, typename toarray>
                static void onorm(const tiarray& iarray, toarray&& oarray)
                {
                        assert(std::isfinite(iarray.minCoeff()));
                        assert(std::isfinite(iarray.maxCoeff()));

                        const auto isum1 = iarray.sum();
                        const auto isum2 = iarray.square().sum();
                        const auto count = static_cast<scalar_t>(iarray.size());
                        const auto imean = isum1 / count;
                        const auto istdv = std::sqrt(isum2 * count - isum1 * isum1) / count;

                        oarray = (iarray - imean) / istdv;

                        assert(std::isfinite(oarray.minCoeff()));
                        assert(std::isfinite(oarray.maxCoeff()));
                }

                template <typename tiarray, typename toarray>
                static void gnorm(tiarray&& iarray, const toarray& oarray)
                {
                        assert(std::isfinite(iarray.minCoeff()));
                        assert(std::isfinite(iarray.maxCoeff()));
                        assert(std::isfinite(oarray.minCoeff()));
                        assert(std::isfinite(oarray.maxCoeff()));

                        const auto isum1 = iarray.sum();
                        const auto isum2 = iarray.square().sum();
                        const auto count = static_cast<scalar_t>(iarray.size());
                        const auto imean = isum1 / count;
                        const auto istdv = std::sqrt(isum2 * count - isum1 * isum1) / count;

                        const auto osum1 = oarray.sum();
                        const auto oisum = (oarray * (iarray - imean)).sum();

                        iarray = oarray / (istdv) -
                                 osum1 / (count * istdv) -
                                 (iarray - imean) * oisum / (count * istdv * istdv * istdv);

                        assert(std::isfinite(iarray.minCoeff()));
                        assert(std::isfinite(iarray.maxCoeff()));
                }

                // attributes
                norm_params_t   m_params;
        };

        template <typename tidata, typename todata>
        void norm4d_t::output(const tidata& idata, todata&& odata) const
        {
                assert(m_params.valid(idata) && m_params.valid(odata));

                const auto count = idata.template size<0>();
                const auto imaps = idata.template size<1>();

                switch (m_params.m_ntype)
                {
                case norm_type::global:
                        for (auto x = 0; x < count; ++ x)
                        {
                                onorm(idata.array(x), odata.array(x));
                        }
                        break;
                case norm_type::plane:
                        for (auto x = 0; x < count; ++ x)
                        {
                                for (auto i = 0; i < imaps; ++ i)
                                {
                                        onorm(idata.array(x, i), odata.array(x, i));
                                }
                        }
                        break;
                }
        }

        template <typename tidata, typename todata>
        void norm4d_t::ginput(tidata&& idata, const todata& odata) const
        {
                assert(m_params.valid(idata) && m_params.valid(odata));

                const auto count = idata.template size<0>();
                const auto imaps = idata.template size<1>();

                switch (m_params.m_ntype)
                {
                case norm_type::global:
                        for (auto x = 0; x < count; ++ x)
                        {
                                gnorm(idata.array(x), odata.array(x));
                        }
                        break;
                case norm_type::plane:
                        for (auto x = 0; x < count; ++ x)
                        {
                                for (auto i = 0; i < imaps; ++ i)
                                {
                                        gnorm(idata.array(x, i), odata.array(x, i));
                                }
                        }
                        break;
                }
        }
}
