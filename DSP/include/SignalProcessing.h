#include <cmath>
#include <iostream>

/* Used to describe domain of a 1D signal in the form of [lb_, ub_). */
class Domain1D
{
private:
    int lb_ = 0;
    int ub_ = 0;

public:
    /* Constructors. */
    Domain1D();
    Domain1D(const int lowerBound, const int upperBound);

    Domain1D(const Domain1D& domain); // copy constructor
    Domain1D(Domain1D&& domain); // move constructor

    /* Assign operators. */
    Domain1D& operator=(const Domain1D& domain);
    Domain1D& operator=(Domain1D&& domain);

    int LowerBound() const;
    int UpperBound() const;

    void SetBounds(const int lb, const int ub);
};

/* A template for 1D signals. The true signal is data_ restricted to domain_. */
template <typename T>
class Signal1D
{
private:
    T* data_ = nullptr;
    int stride_ = 0;
    Domain1D domain_;

public:
    Signal1D() = delete;
    Signal1D(const int len);
    Signal1D(const int len, const Domain1D& domain);
    Signal1D(const int len, Domain1D&& domain);
    Signal1D(Signal1D&& signal);

    ~Signal1D();

    T* GetData();
    int GetStride() const;
    Domain1D GetDomain() const;
};

template <typename T>
T* Signal1D<T>::GetData()
{
    return data_;
}

template <typename T>
int Signal1D<T>::GetStride() const
{
    return stride_;
}

template <typename T>
Domain1D Signal1D<T>::GetDomain() const
{
    return domain_;
}

template <typename T>
Signal1D<T>::Signal1D(const int len) : domain_(0, 0)
{
    data_ = new T[len];
    stride_ = len;
}

template <typename T>
Signal1D<T>::Signal1D(const int len, const Domain1D& domain) : domain_(domain), stride_(len)
{
    data_ = new T[len];
}

template <typename T>
Signal1D<T>::Signal1D(const int len, Domain1D&& domain) : domain_(domain.LowerBound(), domain.UpperBound()), stride_(len)
{
    data_ = new T[len];
}

template <typename T>
Signal1D<T>::Signal1D(Signal1D&& signal) : domain_(signal.GetDomain()), stride_(signal.GetStride())
{
    data_ = signal.GetData();
}

template <typename T>
Signal1D<T>::~Signal1D()
{
    if (!data_) delete[] data_;

#ifdef DEBUG
    std::cout << "Deleted 1D signal." << std::endl;
#endif
}

/* Used to describe domain of a 2D signal in the form of [xl_, xu_), [yl_, yu_). */
class Domain2D
{
private:
    int xl_ = 0;
    int xu_ = 0;
    int yl_ = 0;
    int yu_ = 0;

public:
    Domain2D() = delete;
    Domain2D(const int xLowerBound, const int xUpperBound, const int yLowerBound, const int yUpperBound);

    Domain2D(const Domain2D& domain);
    Domain2D(Domain2D&& domain);

    Domain2D& operator=(const Domain2D& domain);
    Domain2D& operator=(const Domain2D&& domain);

    int LowerBoundX() const;
    int UpperBoundX() const;
    int LowerBoundY() const;
    int UpperBoundY() const;
};

template <typename T>
class Signal2D
{
private:
    T* data_ = nullptr;
    int xStride_ = 0;
    Domain2D domain_;
};