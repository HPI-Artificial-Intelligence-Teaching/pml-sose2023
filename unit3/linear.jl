# Plots for linear algebra
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    plot_mapping(A)

Plots an R² → R² mapping via a unit ball and a triangle
"""
function plot_mapping(A)
    function plot_points(pts, i1, i2)
        p = plot(pts[1,:], pts[2,:], legend=false, linewidth = 3, color = :black, 
                 xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16, 
                 aspect_ratio=:equal)
        plot!([0,pts[1,i1]],[0,pts[2,i1]], arrow=true, linewidth=5, color = :blue)
        plot!([0,pts[1,i2]],[0,pts[2,i2]], arrow=true, linewidth=5, color = :red)
        xlabel!(L"x_1")
        ylabel!(L"x_2")    
        return p
    end

    U, S, V = svd(A,alg = LinearAlgebra.QRIteration())

    i1, i2 = 1, 26
    θs = range(0, stop=2π, length=100)
    X = hcat(sin.(θs), cos.(θs))'
    p1 = plot_points(X, i1, i2)
    p2 = plot_points(V' * X, i1, i2)
    p3 = plot_points(Diagonal(S) * V' * X, i1, i2)
    p4 = plot_points(U * Diagonal(S) * V' * X, i1, i2)
    p5 = plot_points(A * X, i1, i2)

    return (p1,p2,p3,p4,p5)
end



# plot a sample of basis functions
(p1,p2,p3,p4,p5) = plot_mapping([[1 1];[0 1]])
display(p1)
display(p2)
display(p3)
display(p4)
display(p5)
