#ifndef HOST_IMAGE_HPP_CPMV6RNP
#define HOST_IMAGE_HPP_CPMV6RNP

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <cassert>

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

    template <typename PixType, size_t N>
    class HostImageView
    {
        static_assert( N >= 1, "An image has to have at least one dimension." );
        // TODO: add assert once availble in gcc
        //static_assert( std::is_trivially_copyable<PixType>,
                        //"An image must consist of trivially copyable pixels." );

    protected:

        void invalidate()
        {
            std::fill_n(dims_.begin(), N, 0);
            data_ = nullptr; // don't delete
        }

        static size_t multDims(std::array<size_t, N> const& dims)
        {
            return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
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

        HostImageView()
        { 
            invalidate();
        }

        // TODO: enable copying of views
        HostImageView(HostImageView const& other) = delete;
        HostImageView& operator =(HostImageView const& other) = delete;

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
            return multDims(dims_);
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

    };

    template <typename PixType, size_t N>
    class HostImage : private HostImageView<PixType, N>
    {
        static_assert( N >= 1, "An image has to have at least one dimension." );
        // TODO: add assert once availble in gcc
        //static_assert( std::is_trivially_copyable<PixType>,
                        //"An image must consist of trivially copyable pixels." );

        // holds the data, while this object manages its lifetime.
        typedef HostImageView<PixType, N> view_type;

        void dealloc()
        {
            delete[] view_type::data_;
        }

        HostImage(std::array<size_t, N> const& dims, PixType* data)
            : view_type(dims, data)
        { }

        /**
         * Release the raw pointer to the pixel data.
         *
         * @note Please make sure to know the dimensions before invalidating
         * this image and pulling out the data.
         */
        PixType* releaseData() {
            PixType* result = view_type::data_;
            view_type::invalidate();
            return result;
        }

        friend class HostImage<PixType, N-1>;

    public:

        HostImage(std::array<size_t, N> const& dims)
            : view_type(dims, new PixType[view_type::multDims(dims)])
        { }

        HostImage(size_t width, size_t height)
            : view_type({{width, height}}, new PixType[width*height])
        { }

        HostImage(size_t width, size_t height, size_t depth)
            : view_type({{width, height, depth}}, new PixType[width*height*depth])
        { }

        /**
         * Creates a single image of N+1 dimensions,
         * out of M images of dimension N.
         *
         * @note input images are deallocated.
         */
        HostImage(std::vector<HostImage<PixType, N-1>> const& subimages);

        /**
         * Transfer data from one image into another, but reduce
         * dimensionality.
         *
         * i.e. 3D image becomes 2D, where the last dimensions is
         * a collapsing of two- 3x4x5 -> 3x20
         */
        // TODO: update comment
        HostImage(HostImage<PixType, N+1>&& toBeCollapsed)
        {
            // create colapsed dimensions
            std::copy_n(toBeCollapsed.view().dimensions().begin(), N, view_type::dims_.begin());
            view_type::dims_[N-1] *= toBeCollapsed.view().dimensions()[N];

            // transfer image data
            view_type::data_ = toBeCollapsed.releaseData();
        }

        ~HostImage() { dealloc(); }

        HostImage()
        { 
            view_type::invalidate();
        }

        // disable copying because expensive
        HostImage(HostImage const& other) = delete;
        HostImage& operator =(HostImage const& other) = delete;

        // move constructor
        HostImage(HostImage&& other)
            : view_type(std::move(other))
        { }

        // move assignment
        HostImage& operator =(HostImage&& other)
        {
            view_type::operator=(std::move(other));
            //std::copy_n(other.dims_.begin(), N, dims_.begin());
            //pixArray_ = other.pixArray_;

            //other.view_type::invalidate();
            return *this;
        }

        /**
         * Return whether the image is valid
         */
        bool valid()
        {
            return view_type::valid();
        }

        operator bool()
        {
            return valid();
        }

        view_type& view() { return *this; }
        view_type const& view() const { return *this; }

    };

    template <typename PixType, size_t N>
    HostImage<PixType, N>::HostImage(std::vector<HostImage<PixType, N-1>> const& subimages)
    {
        assert(subimages.size() > 0);

        view_type::dims_[N-1] = subimages.size();

        // copy the first image dimensions
        std::array<size_t, N-1> const& otherdims = subimages[0].view().dimensions();
        std::copy(otherdims.begin(), otherdims.end(), view_type::dims_.begin());

        // can now allocate space
        view_type::data_ = new PixType[view().totalSize()];
        PixType* writePtr = view_type::data_; // current write point

        typedef HostImage<PixType, N-1> subimage_type;
        for(subimage_type const& subimage : subimages)
        {
            // ensure all dimensions match
            assert( std::equal( subimage.view().dimensions().begin(),
                                subimage.view().dimensions().end(),
                                view().dimensions().begin() ) );

            writePtr = std::copy( subimage.view().begin(), subimage.view().end(), writePtr );
        }
    }

}

#endif /* end of include guard: HOST_IMAGE_HPP_CPMV6RNP */
