# Plots for vector-valued function derivative
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using LaTeXStrings
using Plots

# helper function to create a quiver grid
meshgrid(x, y) = (repeat(x, outer = length(y)), repeat(y, inner = length(x)))

"""
    plot_derivative(f, df)

Plots
"""
function plot_derivative(f, df; min_x = -2, max_x = 2, min_y = -2, max_y = 2)
    X = range(min_x, stop = max_x, length = 100)
    Y = range(min_y, stop = max_y, length = 100)

    X_mesh = range(min_x, stop = max_x, length = 30)
    Y_mesh = range(min_y, stop = max_y, length = 30)

    xs, ys = meshgrid(
        range(min_x, stop = max_x, length = 11),
        range(min_y, stop = max_y, length = 11),
    )

    p1 = surface(
        X,
        Y,
        f,
        fillalpha = 0.8,
        legend = false,
        camera = (45, 40),
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    wireframe!(X_mesh, Y_mesh, f, c = :blue, alpha = 0.9)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    zlabel!(L"f(x_1,x_2)")

    xlims!(min_x, max_x)
    ylims!(min_y, max_y)

    p2 = contour(
        X,
        Y,
        f,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )


    quiver!(xs, ys, quiver = df, c = :blue, linewidth = 2)
    xlabel!(L"x_1")
    ylabel!(L"x_2")

    xlims!(min_x, max_x)
    ylims!(min_y, max_y)

    return p1, p2
end


# plot a derivative of a 2D->1D function
p1, p2 = plot_derivative((x, y) -> x^3 - 3x + y^2, (x, y) -> [3x^2 - 3; 2y] / 25)
savefig(p1, "~/Downloads/3D.png")
display(p1)
savefig(p2, "~/Downloads/quiver.png")
display(p2)
