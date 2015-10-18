#pragma once

namespace min
{
        ///
        /// \brief stochastic optimization methods
        ///
        enum class stoch_optimizer
        {
                SG,                             ///< stochastic gradient
                SGA,                            ///< stochastic gradient averaging
                SIA,                            ///< stochastic iterate averaging
                AG,                             ///< Nesterov's accelerated gradient
                AGGR,                           ///< Nesterov's accelerated gradient (with gradient check-based restarts)
                ADAGRAD,                        ///< AdaGrad
                ADADELTA                        ///< AdaDelta
        };
}

