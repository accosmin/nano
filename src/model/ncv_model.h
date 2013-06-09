#ifndef NANOCV_MODEL_H
#define NANOCV_MODEL_H

#include "ncv_task.h"
#include "ncv_loss.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        // manage models (register new ones, query and clone them)
        class model_t;
        typedef manager_t<model_t>              model_manager_t;
        typedef model_manager_t::robject_t      rmodel_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // generic model: output = model(input).
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class model_t : public clonable_t<model_t>
        {
        public:

                // constructor
                model_t(const string_t& name, const string_t& description);

                // destructor
                virtual ~model_t() {}

                // train the model
                bool train(const task_t& task, const fold_t& fold, const loss_t& loss);

                // evaluate the model (compute the average loss value & error)
                void test(const task_t& task, const fold_t& fold, const loss_t& loss,
                        scalar_t& lvalue, scalar_t& lerror) const;

                // compute the model output
                virtual vector_t forward(const image_t& image, coord_t x, coord_t y) const = 0;
                vector_t forward(const image_t& image, const rect_t& region) const;

                // save/load from file
                bool save(const string_t& path) const;
                bool load(const string_t& path);

                // access functions
                size_t n_rows() const { return m_rows; }
                size_t n_cols() const { return m_cols; }
                size_t n_outputs() const { return m_outputs; }
                size_t n_parameters() const { return m_parameters; }

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const = 0;
                virtual bool load(boost::archive::binary_iarchive& ia) = 0;

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize() = 0;

                // initialize parameters
                virtual void zero() = 0;
                virtual void random() = 0;

                // train the model
                virtual bool train(const task_t& task, const samples_t& samples, const loss_t& loss) = 0;

        private:

                // attributes
                size_t          m_rows, m_cols;         // input patch size
                size_t          m_outputs;              // output size
                size_t          m_parameters;
        };
}

#endif // NANOCV_MODEL_H
