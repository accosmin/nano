#pragma once

namespace zob
{
        ///
        /// \brief batch optimization methods
        ///
        enum class batch_optimizer
        {
                GD,                             ///< gradient descent
                CGD,                            ///< conjugate gradient descent (default version)
                LBFGS,                          ///< limited-memory BFGS

                CGD_HS,                         ///< various conjugate gradient descent versions
                CGD_FR,
                CGD_PRP,
                CGD_CD,
                CGD_LS,
                CGD_DY,
                CGD_N,
                CGD_DYCD,
                CGD_DYHS
        };
}

