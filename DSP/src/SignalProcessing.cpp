#include "SignalProcessing.h"
#include <iostream>

Domain1D::Domain1D()
{
#ifdef DEBUG
    std::cout << "In " << __FUNCTION__ << std::endl;
#endif
}

Domain1D::Domain1D(const int lowerBound, const int upperBound) : lb_{lowerBound}, ub_{upperBound} {}

Domain1D::Domain1D(const Domain1D& domain)
{
    lb_ = domain.LowerBound();
    ub_ = domain.UpperBound();
}

Domain1D::Domain1D(Domain1D&& domain)
{
#ifdef DEBUG
    std::cout << "In " << __FUNCTION__ << std::endl;
#endif

    lb_ = domain.LowerBound();
    ub_ = domain.UpperBound();
}

Domain1D& Domain1D::operator=(const Domain1D& domain)
{
    lb_ = domain.LowerBound();
    ub_ = domain.UpperBound();

    return *this;
}

Domain1D& Domain1D::operator=(Domain1D&& domain)
{
    lb_ = domain.LowerBound();
    ub_ = domain.UpperBound();

    return *this;
}

int Domain1D::LowerBound() const { return lb_; }

int Domain1D::UpperBound() const { return ub_; }

void Domain1D::SetBounds(const int lb, const int ub)
{
    lb_ = lb;
    ub_ = ub;
}

Domain2D::Domain2D(const int xLowerBound, const int xUpperBound, const int yLowerBound, const int yUpperBound) : 
xl_{xLowerBound}, xu_{xUpperBound}, yl_{yLowerBound}, yu_{yUpperBound} {}

Domain2D::Domain2D(const Domain2D& domain)
{
    xl_ = domain.LowerBoundX();
    xu_ = domain.UpperBoundX();

    yl_ = domain.LowerBoundY();
    yu_ = domain.UpperBoundY();
}

Domain2D::Domain2D(Domain2D&& domain)
{
    xl_ = domain.LowerBoundX();
    xu_ = domain.UpperBoundX();

    yl_ = domain.LowerBoundY();
    yu_ = domain.UpperBoundY();
}

Domain2D& Domain2D::operator=(const Domain2D& domain)
{
    xl_ = domain.LowerBoundX();
    xu_ = domain.UpperBoundX();

    yl_ = domain.LowerBoundY();
    yu_ = domain.UpperBoundY();

    return *this;
}

Domain2D& Domain2D::operator=(const Domain2D&& domain)
{
    xl_ = domain.LowerBoundX();
    xu_ = domain.UpperBoundX();

    yl_ = domain.LowerBoundY();
    yu_ = domain.UpperBoundY();

    return *this;
}

int Domain2D::LowerBoundX() const
{
    return xl_;
}

int Domain2D::UpperBoundX() const
{
    return xu_;
}

int Domain2D::LowerBoundY() const
{
    return yl_;
}

int Domain2D::UpperBoundY() const
{
    return yu_;
}

