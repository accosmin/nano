#pragma once

#include "sampler.h"
#include "text/enum_string.h"

namespace nano
{
        ///
        /// \brief image warping type.
        ///
        enum class warp_type
        {
                translation,            ///<
                rotation,               ///<
                random,                 ///<
                mixed,                  ///< combines translation & rotation
        };

        template <>
        inline std::map<warp_type, std::string> enum_string<warp_type>()
        {
                return
                {
                        { warp_type::translation,       "translation" },
                        { warp_type::rotation,          "rotation" },
                        { warp_type::random,            "random" },
                        { warp_type::mixed,             "mixed" }
                };
        }

        void warp(tensor3d_t&, const warp_type,
                const scalar_t noise, const scalar_t sigma, const scalar_t alpha, const scalar_t beta);

        ///
        /// \brief generate samples by randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct sampler_warp_t final : public sampler_t
        {
                explicit sampler_warp_t(const string_t& configuration = string_t());

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) final;

        private:

                // attributes
                matrix_t        m_gradx;        ///< buffer: horizontal gradient
                matrix_t        m_grady;        ///< buffer: vertical gradient
                matrix_t        m_fieldx;       ///< buffer: horizontal displacement
                matrix_t        m_fieldy;       ///< buffer: vertical displacement
        };
}
