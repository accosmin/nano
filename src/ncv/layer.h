#ifndef NANOCV_LAYER_H
#define NANOCV_LAYER_H

#include "common/manager.hpp"
#include "types.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ncv
{
        class layer_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<layer_t>                      layer_manager_t;
        typedef layer_manager_t::robject_t              rlayer_t;
        typedef std::vector<rlayer_t>                   rlayers_t;

        class ivectorizer_t;
        class ovectorizer_t;

        ///
        /// \brief process a set of inputs of size (irows, icols) and produces a set of outputs of size (orows, ocols)
        ///
        class layer_t : public clonable_t<layer_t>
        {
        public:

                ///
                /// \brief destructor
                ///
                virtual ~layer_t() {}

                //
                ///
                /// \brief resize to process new inputs
                /// \return the number of parameters
                ///
                virtual size_t resize(size_t idims, size_t irows, size_t icols) = 0;

                ///
                /// \brief reset parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief reset parameters to random values in the [min, max] range
                ///
                virtual void random_params(scalar_t min, scalar_t max) = 0;

                ///
                /// \brief serialize parameters
                ///
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const = 0;

                ///
                /// \brief serialize gradients
                ///
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const = 0;

                ///
                /// \brief unserialize parameters
                ///
                virtual ivectorizer_t& load_params(ivectorizer_t& s) = 0;

                ///
                /// \brief compute the output tensor
                ///
                virtual const tensor3d_t& forward(const tensor3d_t& input) const = 0;

                ///
                /// \brief compute the gradient tensor
                ///
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const = 0;

                ///
                /// \brief save to binary file archive
                ///
                virtual bool save(boost::archive::binary_oarchive& oa) const = 0;

                ///
                /// \brief load from binary file archive
                ///
                virtual bool load(boost::archive::binary_iarchive& ia) = 0;

                ///
                /// \brief save its description as an image (if possible)
                ///
                virtual bool save_as_image(const string_t& basepath) const = 0;

                ///
                /// \brief access function
                /// \return the number of input dimensions
                ///
                virtual size_t n_idims() const = 0;

                ///
                /// \brief access function
                /// \return the number of input rows
                ///
                virtual size_t n_irows() const = 0;

                ///
                /// \brief access function
                /// \return the number of input columns
                ///
                virtual size_t n_icols() const = 0;

                ///
                /// \brief access function
                /// \return the number of output dimensions
                ///
                virtual size_t n_odims() const = 0;

                ///
                /// \brief access function
                /// \return the number of output rows
                ///
                virtual size_t n_orows() const = 0;

                ///
                /// \brief access function
                /// \return the number of output columns
                ///
                virtual size_t n_ocols() const = 0;
        };
}

#endif // NANOCV_LAYER_H
