#include "stoch.h"
#include "stoch/ag.h"
#include "stoch/sg.h"
#include "stoch/ngd.h"
#include "stoch/sgm.h"
#include "stoch/adam.h"
#include "stoch/adagrad.h"
#include "stoch/adadelta.h"

namespace nano
{
        state_t minimize(const stoch_params_t& params, const problem_t& problem, const vector_t& x0)
        {
                switch (params.m_optimizer)
                {
                case stoch_optimizer::AG:       return stoch_ag_t(ag_restart::none)(params, problem, x0);
                case stoch_optimizer::AGFR:     return stoch_ag_t(ag_restart::function)(params, problem, x0);
                case stoch_optimizer::AGGR:     return stoch_ag_t(ag_restart::gradient)(params, problem, x0);

                case stoch_optimizer::ADAGRAD:  return stoch_adagrad_t()(params, problem, x0);
                case stoch_optimizer::ADADELTA: return stoch_adadelta_t()(params, problem, x0);
                case stoch_optimizer::ADAM:     return stoch_adam_t()(params, problem, x0);

                case stoch_optimizer::SGM:      return stoch_sgm_t()(params, problem, x0);
                case stoch_optimizer::NGD:      return stoch_ngd_t()(params, problem, x0);
                case stoch_optimizer::SG:
                default:                        return stoch_sg_t()(params, problem, x0);
                }
        }
}

