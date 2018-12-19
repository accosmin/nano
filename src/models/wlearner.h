#pragma once

#include "tensor.h"

namespace nano
{
        // todo: generalize it to use other features (e.g products of inputs, LBPs|HOGs)!

        ///
        /// \brief a weak learner is performing a transformation of the selected feature value.
        ///
        class wlearner_t
        {
        public:

                ///
                /// \brief default constructor
                ///
                wlearner_t() = default;

                ///
                /// \brief change the selected feature
                ///
                auto& feature(const tensor_size_t feature)
                {
                        assert(feature >= 0);
                        m_feature = feature;
                        return *this;
                }

                ///
                /// \brief returns the selected feature
                ///
                auto feature() const { return m_feature; }

        protected:

                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
        };
}
