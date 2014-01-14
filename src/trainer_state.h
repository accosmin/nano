#ifndef NANOCV_TRAINER_STATE_H
#define NANOCV_TRAINER_STATE_H

#include "types.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // track the current/optimum model state.
        /////////////////////////////////////////////////////////////////////////////////////////
                
        struct trainer_state_t
        {
                // constructor
                trainer_state_t(size_t n_parameters)
                        :       m_params(n_parameters),
                                m_tvalue(std::numeric_limits<scalar_t>::max()),
                                m_terror(std::numeric_limits<scalar_t>::max()),
                                m_vvalue(std::numeric_limits<scalar_t>::max()),
                                m_verror(std::numeric_limits<scalar_t>::max())
                {
                }

                // update the current/optimum state with a possible better state
                void update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror,
                            scalar_t vvalue, scalar_t verror)
                {
                        if (verror < m_verror)
                        {
                                m_params = params;
                                m_tvalue = tvalue;
                                m_terror = terror;
                                m_vvalue = vvalue;
                                m_verror = verror;
                        }
                }
                void update(const trainer_state_t& state)
                {
                        update(state.m_params, state.m_tvalue, state.m_terror, state.m_vvalue, state.m_verror);
                }

                // attributes
                vector_t        m_params;       // current model parameters
                scalar_t        m_tvalue;       // train loss value
                scalar_t        m_terror;       // train error
                scalar_t        m_vvalue;       // validation loss value
                scalar_t        m_verror;       // validation error
        };
}

#endif // NANOCV_TRAINER_STATE_H
