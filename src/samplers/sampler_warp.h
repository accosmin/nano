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

        ///
        /// \brief generate samples by randomly warp images, like described in:
        ///      "Training Invariant Support Vector Machines using Selective Sampling", by
        ///      Gaelle Loosli, Stephane Canu & Leon Bottou
        ///
        struct NANO_PUBLIC sampler_warp_t final : public sampler_t
        {
                explicit sampler_warp_t(const string_t& configuration = string_t());

                virtual void get(tensor3d_t&, vector_t*, string_t*) final;

                ///
                /// \brief access the displacement fields as a RGBA image
                ///
                const tensor3d_t& fimage() const { return m_fimage; }

        private:

                // attributes
                tensor3d_t      m_gradx;        ///< buffer: horizontal gradient per plane
                tensor3d_t      m_grady;        ///< buffer: vertical gradient per plane
                matrix_t        m_fieldx;       ///< buffer: horizontal displacement
                matrix_t        m_fieldy;       ///< buffer: vertical displacement
                tensor3d_t      m_fimage;       ///< buffer: displacement fields as RGBA
        };
}
