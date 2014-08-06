#include "criterion.h"
#include "loss.h"
#include "task.h"
#include <cassert>

namespace ncv
{        
        criterion_t::criterion_t(
                const string_t& configuration,
                const string_t& description)
                :       clonable_t<criterion_t>(configuration, description),
                        m_lambda(0.0),
                        m_type(type::value),
                        m_value(0.0),
                        m_error(0.0),
                        m_count(0)
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
                m_params = model.params();
                reset();
                return *this;
        }

        criterion_t& criterion_t::reset(const vector_t& params)
        {
                assert(m_model->psize() == params.size());
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

        void criterion_t::reset()
        {
                m_value = 0.0;
                m_error = 0.0;
                m_vgrad.resize(dimensions());
                m_vgrad.setZero();
                m_count = 0;
        }

        criterion_t& criterion_t::operator+=(const criterion_t& other)
        {
                m_value += other.m_value;
                m_vgrad += other.m_vgrad;
                m_error += other.m_error;
                m_count += other.m_count;
                return *this;
        }

        void criterion_t::cumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                // loss value
                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
                
                // loss gradient
                switch (m_type)
                {
                case type::value:
                        break;

                case type::vgrad:
                        m_vgrad += m_model->pgrad(loss.vgrad(target, output));
                        break;
                }
        }
        
        void criterion_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(sample.m_index < task.n_images());
                
                const image_t& image = task.image(sample.m_index);
                const vector_t& target = sample.m_target;                
                const vector_t& output = m_model->output(image, sample.m_region).vector();
                
                cumulate(output, target, loss);
        }
        
        void criterion_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t& output = m_model->output(input).vector();
                
                cumulate(output, target, loss);
        }
        
        void criterion_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t& output = m_model->output(input).vector();
                
                cumulate(output, target, loss);
        }
        
        scalar_t criterion_t::value() const
        {
                assert(count() > 0);

                return m_value / count();
        }

        scalar_t criterion_t::error() const
        {
                assert(count() > 0);

                return m_error / count();
        }

        vector_t criterion_t::vgrad() const
        {
                assert(count() > 0);

                return m_vgrad / count();
        }

        size_t criterion_t::count() const
        {
                return m_count;
        }

        size_t criterion_t::dimensions() const
        {
                return static_cast<size_t>(m_params.size());
        }

        scalar_t criterion_t::lambda() const
        {
                return m_lambda;
        }
}
	
