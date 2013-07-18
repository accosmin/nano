#ifndef NANOCV_OLD_MODEL_H
#define NANOCV_OLD_MODEL_H

#include "task/task.h"
#include "loss/loss.h"
#include "core/tensor3d.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
//        // manage models (register new ones, query and clone them)
//        class model_t;
//        typedef manager_t<model_t>              model_manager_t;
//        typedef model_manager_t::robject_t      rmodel_t;

//        /////////////////////////////////////////////////////////////////////////////////////////
//        // generic model: output = model(input).
//        /////////////////////////////////////////////////////////////////////////////////////////
                
//        class model_t : public clonable_t<model_t>
//        {
//        public:

//                // constructor
//                model_t(const string_t& description);

//                // destructor
//                virtual ~model_t() {}

//                // train the model
//                bool train(const task_t& task, const fold_t& fold, const loss_t& loss, optimizer trainer);

//                // evaluate the model (compute the average loss value & error)
//                void test(const task_t& task, const fold_t& fold, const loss_t& loss,
//                        scalar_t& lvalue, scalar_t& lerror) const;

//                // compute the model output
//                vector_t process(const image_t& image, coord_t x, coord_t y) const;
//                vector_t process(const image_t& image, const rect_t& region) const;
//                virtual vector_t process(const tensor3d_t& input) const = 0;

//                // save/load from file
//                bool save(const string_t& path) const;
//                bool load(const string_t& path);

//                // access functions
//                size_t n_rows() const { return m_rows; }
//                size_t n_cols() const { return m_cols; }
//                size_t n_inputs() const;
//                size_t n_outputs() const { return m_outputs; }
//                size_t n_parameters() const { return m_parameters; }
//                color_mode color() const { return m_color; }

//        protected:

//                struct data_t
//                {
//                        data_t(const task_t& task,
//                               const samples_t& samples)
//                                :       m_task(task),
//                                        m_samples(samples)
//                        {
//                        }

//                        const task_t&           m_task;
//                        const samples_t&        m_samples;
//                        std::vector<size_t>     m_indices;
//                };

//                // compose the input data
//                tensor3d_t make_input(const image_t& image, coord_t x, coord_t y) const;
//                tensor3d_t make_input(const image_t& image, const rect_t& region) const;

//                // save/load from file
//                virtual bool save(boost::archive::binary_oarchive& oa) const = 0;
//                virtual bool load(boost::archive::binary_iarchive& ia) = 0;

//                // save/load from parameter vector
//                virtual bool save(vector_t& x) const = 0;
//                virtual bool load(const vector_t& x) = 0;

//                // resize to new inputs/outputs, returns the number of parameters
//                virtual size_t resize() = 0;

//                // initialize parameters
//                virtual void zero() = 0;
//                virtual void random() = 0;

//                // construct the list of valid training samples
//                virtual void prune(data_t& data) const = 0;

//                // compute loss value & gradient (given current
//                virtual scalar_t value(const data_t& data, const loss_t& loss) const = 0;
//                virtual scalar_t vgrad(const data_t& data, const loss_t& loss, vector_t& grad) const = 0;

//        private:

//                // train the model
//                bool train_batch(const data_t& data, const loss_t& loss, optimizer trainer);
//                bool train_stochastic(const data_t& data, const loss_t& loss);

//        private:

//                // attributes
//                size_t          m_rows, m_cols;         // input patch size
//                size_t          m_outputs;              // output size
//                size_t          m_parameters;
//                color_mode      m_color;                // input color mode
//        };
}

#endif // NANOCV_OLD_MODEL_H
