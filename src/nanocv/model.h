#ifndef NANOCV_MODEL_H
#define NANOCV_MODEL_H

#include "common/manager.hpp"
#include "image.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        class model_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<model_t>              model_manager_t;
        typedef model_manager_t::robject_t      rmodel_t;

        class task_t;

        ///
        /// \brief generic model used for computing:
        ///     - the output for an image patch
        //      - its parameters gradient
        ///
        class model_t : public clonable_t<model_t>
        {
        public:

                model_t(const string_t& parameters, const string_t& description);

                ///
                /// \brief constructor
                ///
                model_t();

                ///
                /// \brief destructor
                ///
                virtual ~model_t() {}

                ///
                /// \brief resize to process new inputs
                ///
                bool resize(size_t rows, size_t cols, size_t outputs, color_mode color, bool verbose);

                ///
                /// \brief resize to process new inputs compatible with the given task
                ///
                bool resize(const task_t& task, bool verbose);

                ///
                /// \brief compute the model's output
                ///
                const tensor_t& forward(const image_t& image, coord_t x, coord_t y) const;
                const tensor_t& forward(const image_t& image, const rect_t& region) const;
                const tensor_t& forward(const vector_t& input) const;
                virtual const tensor_t& forward(const tensor_t& input) const = 0;

                ///
                /// \brief save its parameters to file
                ///
                bool save(const string_t& path) const;

                ///
                /// \brief load its parameters from file
                ///
                bool load(const string_t& path);

                ///
                /// \brief load its parameters from vector
                ///
                virtual bool load_params(const vector_t& x) = 0;

                ///
                /// \brief set parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief set parameters to random values
                ///
                virtual void random_params() = 0;

                ///
                /// \brief current parameters
                ///
                virtual vector_t params() const = 0;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual vector_t gradient(const vector_t& output) const = 0;

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor_t& backward(const vector_t& output) const = 0;

                ///
                /// \brief construct (from a random initialization) an input that matches closely the target
                ///
                tensor_t generate(const vector_t& target) const;

                // access functions
                size_t irows() const { return m_rows; }
                size_t icols() const { return m_cols; }
                size_t idims() const;
                size_t isize() const { return idims() * irows() * icols(); }
                size_t osize() const { return m_outputs; }
                size_t psize() const { return m_nparams; }
                color_mode color() const { return m_color; }

        protected:

                ///
                /// \brief compose the input data
                ///
                tensor_t make_input(const image_t& image, coord_t x, coord_t y) const;
                tensor_t make_input(const image_t& image, const rect_t& region) const;

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const = 0;
                virtual bool load(boost::archive::binary_iarchive& ia) = 0;

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize(bool verbose) = 0;

        private:

                // attributes
                size_t          m_rows, m_cols;         ///< input patch size
                size_t          m_outputs;              ///< output size
                size_t          m_nparams;              ///< #number of parameters
                color_mode      m_color;                ///< input color mode
        };
}

#endif // NANOCV_MODEL_H
