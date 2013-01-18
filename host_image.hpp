#ifndef HOST_IMAGE_HPP_CPMV6RNP
#define HOST_IMAGE_HPP_CPMV6RNP

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <cassert>
#include <memory>

#include "utils.h"

namespace DynamiCL
{

    /**
     * A simple RGBA pixel, of a particular component type
     */
    template <typename ComponentType>
    union RGBA
    {
        typedef ComponentType component_type;
        typedef component_type* iterator;
        typedef component_type const* const_iterator;

        struct
        {
            ComponentType r;
            ComponentType g;
            ComponentType b;
            ComponentType a;
        };
        component_type components[4];

        component_type&       operator[]( size_t i )       { return components[i]; }
        component_type const& operator[]( size_t i ) const { return components[i]; }
    };

    namespace detail
    {

        template <size_t N>
        static size_t multDims(std::array<size_t, N> const& dims)
        {
            return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        }

    }


    // TODO: keep hostimage alive while references exist?
    /**
     * Represents a "View" onto a contiguous chunk of memory,
     * as an N dimensional image/buffer of PixType structures.
     *
     * Does not manage the pointed-to memory, and allows conversions to other
     * views- such as subimages of an image array.
     */
    template <typename PixType, size_t N>
    class HostImageView
    {
        static_assert( N >= 1, "An image has to have at least one dimension." );
        // TODO: add assert once availble in gcc
        //static_assert( std::is_trivially_copyable<PixType>,
                        //"An image must consist of trivially copyable pixels." );
                        
        // TODO: can this be made constexpr?

    protected:

        void invalidate()
        {
            std::fill_n(dims_.begin(), N, 0);
            data_ = nullptr; // don't delete
        }

        std::array<size_t, N> dims_;
        PixType* data_;

    public:
        typedef PixType* iterator;
        typedef PixType const* const_iterator;
        typedef PixType pixel_type;

        HostImageView(std::array<size_t, N> const& dims, PixType* data)
            : dims_(dims),
              data_(data)
        { }

        template <typename SizeIt>
        HostImageView(SizeIt first, SizeIt last, PixType* data)
            : data_(data)
        { 
            // make sure amount of dimensions supplied is equivalent
            assert(std::distance(first, last) == N);

            std::copy(first, last, dims_.begin());
        }

        HostImageView()
        { 
            invalidate();
        }

        // TODO: enable copying of views
        HostImageView(HostImageView const& other)
            : dims_(other.dims_),
              data_(other.data_)
        { }

        HostImageView& operator =(HostImageView const& other)
        {
            dims_ = other.dims_;
            data_ = other.data_;
        }

        // TODO: explicit copy
        HostImageView copy() const { return HostImageView(dims_, data_); }

        // move constructor
        HostImageView(HostImageView&& other)
            : dims_(std::move(other.dims_)),
              data_(other.data_)
        {
            other.invalidate();
        }

        // move assignment
        HostImageView& operator =(HostImageView&& other)
        {
            std::copy_n(other.dims_.begin(), N, dims_.begin());
            data_ = other.data_;

            other.invalidate();
            return *this;
        }

        size_t width() const { return dims_[0]; }
        size_t height() const { return dims_[1]; }
        size_t depth() const { return dims_[2]; }

        std::array<size_t, N> const& dimensions() const { return dims_; }

        /**
         * Return the total number of pixels in the image.
         */
        size_t totalSize() const
        {
            return detail::multDims(dims_);
        }

        /**
         * Return whether the image is valid
         */
        bool valid()
        {
            // check dimensions
            for (size_t n = 0; n < N; ++n)
            {
                if (dims_[n] == 0)
                {
                    return false;
                }
            }
            // check nullptr
            return data_ != nullptr;
        }

        operator bool()
        {
            return valid();
        }

        // iterators

        iterator begin() { return data_; }
        iterator end()   { return data_ + totalSize(); }

        const_iterator begin() const { return data_; }
        const_iterator end()   const { return data_ + totalSize(); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend()   const { return end(); }

        void const* rawData() const { return static_cast<void const*>(begin()); }
        void*       rawData()       { return static_cast<void*>(begin()); }

        HostImageView<PixType, N-1> operator [](size_t index)
        {
            // calculate dimension of subimage
            std::array<size_t, N-1> dims;
            std::copy_n(dims_.begin(), N-1, dims.begin());
            size_t subimageSize = detail::multDims(dims);

            return HostImageView<PixType, N-1>(dims, data_ + (index*subimageSize));
        }


    };

    /**
     * Manages an N-dimensional buffer of PixType objects in memory,
     * which can be accessed through a HostImageView.
     */
    template <typename PixType, size_t N>
    class HostImage
    {
        static_assert( N >= 1, "An image has to have at least one dimension." );
        // TODO: add assert once availble in gcc
        //static_assert( std::is_trivially_copyable<PixType>,
                        //"An image must consist of trivially copyable pixels." );

        // holds the data, while this object manages its lifetime.
        typedef HostImageView<PixType, N> view_type;
        typedef std::array<size_t, N> dim_type;

        friend class HostImage<PixType, N-1>;
        
        void invalidate()
        {
            std::fill_n(dims_.begin(), N, 0);
            alignedData_ = nullptr;
        }

        // TODO: look into how to optimize returning a view
        dim_type dims_;
        array_ptr<PixType, 256> alignedData_;

    public:

        HostImage(std::array<size_t, N> const& dims)
            : dims_(dims),
              alignedData_(detail::multDims(dims))
        { }

        HostImage(size_t width, size_t height)
            : HostImage(std::array<size_t, 2>{{width, height}})
        { }

        HostImage(size_t width, size_t height, size_t depth)
            : HostImage(std::array<size_t, 3>{{width, height, depth}})
        { }

        /**
         * Creates a single image of N+1 dimensions,
         * out of M images of dimension N.
         *
         * @note input images are deallocated.
         */
        HostImage(std::vector<HostImageView<PixType, N-1>> const& subimages);

        /**
         * Transfer data from one image into a new one, but reduce
         * dimensionality.
         *
         * i.e. 3D image becomes 2D, where the last dimensions is
         * a collapsing of two- 3x4x5 -> 3x20
         */
        HostImage(HostImage<PixType, N+1>&& toBeCollapsed)
        {
            // create colapsed dimensions
            std::copy_n(toBeCollapsed.view().dimensions().begin(), N, dims_.begin());
            dims_[N-1] *= toBeCollapsed.view().dimensions()[N];

            // transfer image data
            alignedData_ = std::move(toBeCollapsed.alignedData_);
            toBeCollapsed.invalidate();
        }

        ~HostImage() { }

        HostImage() { }

        // disable copying because expensive
        HostImage(HostImage const& other) = delete;
        HostImage& operator =(HostImage const& other) = delete;

        // move constructor
        HostImage(HostImage&& other)
            : dims_(std::move(other.dims_)),
              alignedData_(std::move(other.alignedData_))
        {
            other.invalidate();
        }

        // move assignment
        HostImage& operator =(HostImage&& other)
        {
            dims_ = std::move(other.dims_);
            alignedData_ = std::move(other.alignedData_);
            other.invalidate();
            return *this;
        }

        /**
         * Return whether the image is valid
         */
        bool valid()
        {
            // check dimensions
            for (size_t n = 0; n < N; ++n)
            {
                if (dims_[n] == 0)
                {
                    return false;
                }
            }
            // check nullptr
            return alignedData_.ptr() != nullptr;
        }

        operator bool()
        {
            return valid();
        }

        /**
         * Return a view of the memory buffer.
         */
        view_type view()
        {
            // TODO: minimize use of this
            return view_type(dims_, alignedData_.ptr());
        }

        view_type const view() const
        {
            return view_type(dims_, alignedData_.ptr());
        }

    };

    template <typename PixType, size_t N>
    HostImage<PixType, N>::HostImage(std::vector<HostImageView<PixType, N-1>> const& subimages)
    {
        assert(subimages.size() > 0);

        dims_[N-1] = subimages.size();

        // copy the first image dimensions
        std::array<size_t, N-1> const& otherdims = subimages[0].dimensions();
        std::copy(otherdims.begin(), otherdims.end(), dims_.begin());

        // can now allocate space
        // TODO: fix this to be aligned!!!
        alignedData_ = array_ptr<PixType, 256>(detail::multDims(dims_));
        PixType* writePtr = alignedData_.begin(); // current write point

        typedef HostImageView<PixType, N-1> subimage_type;
        for(subimage_type const& subimage : subimages)
        {
            // ensure all dimensions match
            assert( std::equal( subimage.dimensions().begin(),
                                subimage.dimensions().end(),
                                view().dimensions().begin() ) );

            writePtr = std::copy( subimage.begin(), subimage.end(), writePtr );
        }
    }

}

#endif /* end of include guard: HOST_IMAGE_HPP_CPMV6RNP */
