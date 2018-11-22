#pragma once

#include "tensor.h"

namespace nano
{
        // todo: generalize stump_t to use other features (e.g. Haar, HoG)

        ///
        /// \brief a stump is a weak learner that compares the value of a selected feature with a threshold:
        ///     stump(x) = outputs(0) if x(feature) < threshold else v1(0)
        ///
        struct stump_t
        {
                stump_t() = default;

                ///
                /// \brief compute the output/prediction given a 3D tensor input
                ///
                template <typename ttensor3d>
                auto output(const ttensor3d& input) const
                {
                        const auto oindex = input(m_feature) < m_threshold ? 0 : 1;
                        return m_outputs.array(oindex);
                }

                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
                scalar_t        m_threshold{0}; ///< threshold
                tensor4d_t      m_outputs;      ///< (2, #outputs) - predictions below and above the threshold
        };

        using stumps_t = std::vector<stump_t>;
}
