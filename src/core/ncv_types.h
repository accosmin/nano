#ifndef NANOCV_TYPES_H
#define NANOCV_TYPES_H

#include <functional>
#include <string>
#include <vector>
#include <map>
#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>

namespace ncv
{
        // vector
        template
        <
                typename tvalue
        >
        struct vector
        {
                typedef tvalue                                  type_t;

                typedef Eigen::Matrix
                <
                        tvalue,
                        Eigen::Dynamic,
                        1,
                        Eigen::ColMajor
                >                                               vector_t;
                typedef std::vector<vector_t>                   vectors_t;
                typedef typename vectors_t::const_iterator      vectors_const_it;
                typedef typename vectors_t::iterator            vectors_it;
        };

        // matrix
        template
        <
                typename tvalue
        >
        struct matrix
        {
                typedef tvalue                                  type_t;

                typedef Eigen::Matrix
                <       tvalue,
                        Eigen::Dynamic,
                        Eigen::Dynamic,
                        Eigen::RowMajor
                >                                               matrix_t;
                typedef std::vector<matrix_t>                   matrices_t;
                typedef typename matrices_t::const_iterator     matrices_const_it;
                typedef typename matrices_t::iterator           matrices_it;
                typedef typename matrix_t::Index                index_t;
        };

        // numerical types
        typedef std::size_t                     size_t;
        typedef std::size_t                     index_t;
        typedef std::size_t                     count_t;
        typedef std::vector<index_t>            indices_t;
        typedef std::vector<count_t>            counts_t;

        typedef double                          scalar_t;
        typedef std::vector<scalar_t>           scalars_t;

        typedef vector<scalar_t>::vector_t      vector_t;
        typedef vector<scalar_t>::vectors_t     vectors_t;

        typedef matrix<scalar_t>::matrix_t      matrix_t;
        typedef matrix<scalar_t>::matrices_t    matrices_t;

        // strings
        typedef std::string                     string_t;
        typedef std::vector<string_t>           strings_t;
        typedef std::map<string_t, string_t>    string_map_t;

        // lambda
        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;

        // alignment options
        enum class align : int
        {
                left,
                center,
                right
        };

        // machine learning protocol
        enum class protocol : int
        {
                train = 0,            // training
                test                  // testing
        };
}

// serialize matrices and vectors
namespace boost
{
        namespace serialization
        {
                template
                <
                        class tarchive,
                        class tvalue,
                        int Rows,
                        int Cols,
                        int Options
                >
                void serialize(tarchive& ar, Eigen::Matrix<tvalue, Rows, Cols, Options>& mat, const unsigned int)
                {
                        if (tarchive::is_saving::value)
                        {
                                int rows = mat.rows(), cols = mat.cols();
                                ar & rows; ar & cols;

                                for (int i = 0; i < mat.size(); i ++)
                                {
                                        ar & mat(i);
                                }
                        }

                        else
                        {
                                int rows = 0, cols = 0;
                                ar & rows; ar & cols;

                                mat.resize(rows, cols);
                                for (int i = 0; i < mat.size(); i ++)
                                {
                                        ar & mat(i);
                                }
                        }
                }
        }
}

#endif // NANOCV_TYPES_H

