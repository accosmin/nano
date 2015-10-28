#pragma once

#include "string.h"
#include "vision/image.h"
#include "util/manager.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace cortex
{
        class task_t;
        class model_t;

        ///
        /// \brief stores registered prototypes
        ///
        using model_manager_t = manager_t<model_t>;
        using rmodel_t = model_manager_t::trobject;

        NANOCV_PUBLIC model_manager_t& get_models();

        ///
        /// \brief generic model used for computing:
        ///     - the output for an image patch
        //      - its parameters gradient
        ///
        class NANOCV_PUBLIC model_t : public clonable_t<model_t>
        {
        public:

                explicit model_t(const string_t& parameters);

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
                bool resize(const tensor_size_t rows, const tensor_size_t cols, const tensor_size_t outputs,
                            const color_mode color,
                            const bool verbose);

                ///
                /// \brief resize to process new inputs compatible with the given task
                ///
                bool resize(const task_t& task, bool verbose);

                ///
                /// \brief compute the model's output
                ///
                const tensor_t& output(const image_t& image, coord_t x, coord_t y) const;
                const tensor_t& output(const image_t& image, const rect_t& region) const;
                const tensor_t& output(const vector_t& input) const;
                virtual const tensor_t& output(const tensor_t& input) const = 0;

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
                /// \brief save its parameters to vector
                ///
                virtual bool save_params(vector_t& x) const = 0;

                ///
                /// \brief set parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief set parameters to random values
                ///
                virtual void random_params() = 0;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual vector_t gparam(const vector_t& output) const = 0;

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual tensor_t ginput(const vector_t& output) const = 0;

                // access functions
                tensor_size_t irows() const { return m_rows; }
                tensor_size_t icols() const { return m_cols; }
                tensor_size_t idims() const;
                tensor_size_t isize() const { return idims() * irows() * icols(); }
                tensor_size_t osize() const { return m_outputs; }
                virtual tensor_size_t psize() const = 0;
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
                virtual tensor_size_t resize(bool verbose) = 0;

        private:

                // attributes
                tensor_size_t   m_rows, m_cols;         ///< input patch size
                tensor_size_t   m_outputs;              ///< output size
                color_mode      m_color;                ///< input color mode
        };
}

