#include "criterion.h"
#include "task.h"
#include "loss.h"
#include <cassert>

namespace ncv
{        
        criterion_t::criterion_t(const string_t& configuration)
                :       clonable_t<criterion_t>(configuration),
                        m_lambda(0.0),
                        m_type(type::value)
        {
        }

        criterion_t& criterion_t::reset(const rmodel_t& rmodel)
        {
                assert(rmodel);
                return reset(*rmodel);
        }

        criterion_t& criterion_t::reset(const model_t& model)
        {
                m_model = model.clone();
                m_model->save_params(m_params);
                reset();
                return *this;
        }

        criterion_t& criterion_t::reset(const vector_t& params)
        {
                assert(m_model->psize() == static_cast<size_t>(params.size()));
                m_model->load_params(params);
                m_params = params;
                reset();
                return *this;
        }

        criterion_t& criterion_t::reset(type t)
        {
                m_type = t;
                return *this;
        }

        criterion_t& criterion_t::reset(scalar_t lambda)
        {
                m_lambda = lambda;
                return *this;
        }
        
        void criterion_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(sample.m_index < task.n_images());
                
                const image_t& image = task.image(sample.m_index);
                const vector_t& target = sample.m_target;                
                const vector_t& output = m_model->output(image, sample.m_region).vector();

                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                accumulate(output, target, loss, sample.m_weight);
        }
        
        void criterion_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss, scalar_t weight)
        {
                const vector_t& output = m_model->output(input).vector();

                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                accumulate(output, target, loss, weight);
        }
        
        void criterion_t::update(const vector_t& input, const vector_t& target, const loss_t& loss, scalar_t weight)
        {
                const vector_t& output = m_model->output(input).vector();

                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                accumulate(output, target, loss, weight);
        }

        void criterion_t::accumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss, scalar_t weight)
        {
                switch (m_type)
                {
                case type::value:
                        accumulate(weight * loss.value(target, output),
                                   weight * loss.error(target, output));
                        break;

                case type::vgrad:
                        accumulate(weight * m_model->pgrad(loss.vgrad(target, output)),
                                   weight * loss.value(target, output),
                                   weight * loss.error(target, output));
                        break;
                }
        }

        size_t criterion_t::psize() const
        {
                return static_cast<size_t>(m_params.size());
        }

        scalar_t criterion_t::lambda() const
        {
                return m_lambda;
        }
}
	
