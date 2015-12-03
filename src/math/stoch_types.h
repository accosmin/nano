#pragma once

namespace math
{
        ///
        /// \brief stochastic optimization methods
        ///
        enum class stoch_optimizer
        {
                SG,                             ///< stochastic gradient
                SGM,                            ///< stochastic gradient with momentum
                SGA,                            ///< stochastic gradient averaging
                SIA,                            ///< stochastic iterate averaging
                AG,                             ///< Nesterov's accelerated gradient
                AGFR,                           ///< Nesterov's accelerated gradient (with function value-based restarts)
                AGGR,                           ///< Nesterov's accelerated gradient (with gradient-based restarts)
                ADAGRAD,                        ///< AdaGrad
                ADADELTA                        ///< AdaDelta
        };
}

