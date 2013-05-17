#ifndef NANOCV_TYPES_H
#define NANOCV_TYPES_H

#include <functional>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <boost/serialization/vector.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/box.hpp>

namespace ncv
{
        // vector
        template
        <
                typename tvalue
        >
        struct tvector
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

        // fixed size vector
        template
        <
                typename tvalue,
                std::size_t trows
        >
        struct tfixed_size_vector
        {
                typedef tvalue                                  type_t;

                typedef Eigen::Matrix
                <
                        tvalue,
                        trows,
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
        struct tmatrix
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

        // fixed size matrix
        template
        <
                typename tvalue,
                std::size_t trows,
                std::size_t tcols
        >
        struct tfixed_size_matrix
        {
                typedef tvalue                                  type_t;

                typedef Eigen::Matrix
                <       tvalue,
                        trows,
                        tcols,
                        Eigen::RowMajor
                >                                               matrix_t;
                typedef std::vector<matrix_t>                   matrices_t;
                typedef typename matrices_t::const_iterator     matrices_const_it;
                typedef typename matrices_t::iterator           matrices_it;
                typedef typename matrix_t::Index                index_t;
        };

        // numerical types
        typedef std::size_t                     size_t;

        typedef double                          scalar_t;
        typedef std::vector<scalar_t>           scalars_t;

        typedef tvector<scalar_t>::vector_t     vector_t;
        typedef tvector<scalar_t>::vectors_t    vectors_t;

        typedef tmatrix<scalar_t>::matrix_t     matrix_t;
        typedef tmatrix<scalar_t>::matrices_t   matrices_t;

        // pixel geometry
        namespace bgm = boost::geometry::model;
        typedef int                             coord_t;
        typedef bgm::d2::point_xy<coord_t>      point_t;
        typedef bgm::box<point_t>               rect_t;

        inline point_t make_point(coord_t x = 0, coord_t y = 0)
        {
                return point_t(x, y);
        }
        inline rect_t make_rect(coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
        {
                return rect_t(make_point(x, y), make_point(x + w, y + h));
        }
        inline rect_t make_size(coord_t w = 0, coord_t h = 0)
        {
                return make_rect(0, 0, w, h);
        }

        inline coord_t get_left(const rect_t& rect)     { return rect.min_corner().x(); }
        inline coord_t get_right(const rect_t& rect)    { return rect.max_corner().x(); }
        inline coord_t get_top(const rect_t& rect)      { return rect.min_corner().y(); }
        inline coord_t get_bottom(const rect_t& rect)   { return rect.max_corner().y(); }

        inline coord_t get_width(const rect_t& rect)    { return get_right(rect) - get_left(rect); }
        inline coord_t get_height(const rect_t& rect)   { return get_bottom(rect) - get_top(rect); }
        inline coord_t get_rows(const rect_t& rect)     { return get_height(rect); }
        inline coord_t get_cols(const rect_t& rect)     { return get_width(rect); }

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

