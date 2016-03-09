#pragma once

namespace zob
{
        ///
        /// \brief soft-max approximation over the given vector's dimensions.
        /// \param beta approximation constant (the greater the better the approximation)
        ///
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar
        >
        auto softmax_value(tvector&& vector, const tscalar beta = tscalar(10))
        {
                const auto ibeta = tscalar(1) / beta;
                return ibeta * std::log((vector.array() * beta).exp().sum());
        }

        ///
        /// \brief gradient of the soft-max approximation.
        /// \param beta approximation constant (the greater the better the approximation)
        ///
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar
        >
        auto softmax_vgrad(tvector&& vector, const tscalar beta = tscalar(10))
        {
                return (vector.array() * beta).exp() / (vector.array() * beta).exp().sum();
        }
}

