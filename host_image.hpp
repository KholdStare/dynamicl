#ifndef HOST_IMAGE_HPP_CPMV6RNP
#define HOST_IMAGE_HPP_CPMV6RNP

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <cassert>

namespace DynamiCL
{

    template <typename PixType, size_t N>
    class HostImage
    {
        static_assert( N >= 1, "An image has to have at least one dimension." );
        // TODO: add assert once availble in gcc
        //static_assert( std::is_trivially_copyable<PixType>,
                        //"An image must consist of trivially copyable pixels." );

        std::array<size_t, N> dims_;
        PixType* pixArray_;

        void dealloc()
        {
            delete[] pixArray_;
        }

        void invalidate()
        {
            std::fill_n(dims_.begin(), N, 0);
            pixArray_ = nullptr; // don't delete
        }

    public:
        typedef PixType* iterator;
        typedef PixType const* const_iterator;
        typedef PixType pixel_type;

        HostImage(std::array<size_t, N> dims)
            : dims_(dims),
              pixArray_(new PixType[totalSize()])
        { }

        HostImage(size_t width, size_t height)
            : dims_({{width, height}}),
              pixArray_(new PixType[width*height])
        { }

        HostImage(std::array<size_t, N> dims, PixType* data)
            : dims_(dims),
              pixArray_(data)
        { }

        HostImage(size_t width, size_t height, size_t depth)
            : dims_({{width, height, depth}}),
              pixArray_(new PixType[width*height*depth])
        { }

        /**
         * Creates a single image of N+1 dimensions,
         * out of M images of dimension N.
         *
         * @note input images are deallocated.
         */
        HostImage(std::vector<HostImage<PixType, N-1>> const& subimages);

        ~HostImage() { dealloc(); }

        HostImage()
        { 
            invalidate();
        }

        // disable copying because expensive
        HostImage(HostImage const& other) = delete;
        HostImage& operator =(HostImage const& other) = delete;

        // move constructor
        HostImage(HostImage&& other)
            : dims_(std::move(other.dims_)),
              pixArray_(other.pixArray_)
        {
            other.invalidate();
        }

        // move assignment
        HostImage& operator =(HostImage&& other)
        {
            std::copy_n(other.dims_.begin(), N, dims_.begin());
            pixArray_ = other.pixArray_;

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
            return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<size_t>());
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
            return pixArray_ != nullptr;
        }

        operator bool()
        {
            return valid();
        }

        // iterators

        iterator begin() { return pixArray_; }
        iterator end()   { return pixArray_ + totalSize(); }

        const_iterator begin() const { return pixArray_; }
        const_iterator end()   const { return pixArray_ + totalSize(); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend()   const { return end(); }

        void const* rawData() const { return static_cast<void const*>(begin()); }
        void*       rawData()       { return static_cast<void*>(begin()); }

        /**
         * Release the raw pointer to the pixel data.
         *
         * @note Please make sure to know the dimensions before invalidating
         * this image and pulling out the data.
         */
        iterator releaseData() {
            iterator result = pixArray_;
            invalidate();
            return result;
        }
    };

    template <typename PixType, size_t N>
    HostImage<PixType, N>::HostImage(std::vector<HostImage<PixType, N-1>> const& subimages)
        : pixArray_(nullptr)
    {

        assert(subimages.size() > 0);

        dims_[N-1] = subimages.size();

        // copy the first image dimensions
        std::array<size_t, N-1> const& otherdims = subimages[0].dimensions();
        std::copy(otherdims.begin(), otherdims.end(), dims_.begin());

        // can now allocate space
        pixArray_ = new PixType[totalSize()];
        PixType* writePtr = pixArray_; // current write point

        typedef HostImage<PixType, N-1> subimage_type;
        for(subimage_type const& subimage : subimages)
        {
            // ensure all dimensions match
            assert( std::equal( subimage.dimensions().begin(),
                                subimage.dimensions().end(),
                                dims_.begin() ) );

            writePtr = std::copy( subimage.begin(), subimage.end(), writePtr );
        }
    }

    /**
     * Transfer data from one image into another, but reduce
     * dimensionality.
     *
     * i.e. 3D image becomes 2D, where the last dimensions is
     * a collapsing of two- 3x4x5 -> 3x20
     */
    template <typename PixType, size_t N>
    HostImage<PixType, N-1>
    collapseDimension(HostImage<PixType, N>& image)
    {
        // create colapsed dimensions
        std::array<size_t, N-1> dims;
        std::copy_n(image.dimensions().begin(), N-1, dims.begin());
        dims[N-2] *= image.dimensions()[N-1];

        // transfer image data
        return HostImage<PixType, N-1>(dims, image.releaseData());
    }

}

#endif /* end of include guard: HOST_IMAGE_HPP_CPMV6RNP */
