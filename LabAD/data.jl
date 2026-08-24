# Moon example, see https://juliaai.github.io/MLJBase.jl/stable/datasets/#MLJBase.make_moons
import MLJBase, Tables
function random_moon(num_data; noise = 0.1)
    X_table, y_cat = MLJBase.make_moons(num_data, noise = noise)
    X = Tables.matrix(X_table)
    y = 2(float.(y_cat.refs) .- 1.5)
    return X, y
end

# Utility to plot the dataset and the model outputs
using Plots, Colors
function plot_moon(model, W, X, y)
	col = [Colors.JULIA_LOGO_COLORS.red, Colors.JULIA_LOGO_COLORS.blue]
	scatter(X[:, 1], X[:, 2], markerstrokewidth=0, color = col[round.(Int, (3 .+ y) / 2)], label = "")
    x1 = range(minimum(X[:, 1]), stop = maximum(X[:, 1]), length = 30)
    x2 = range(minimum(X[:, 2]), stop = maximum(X[:, 2]), length = 30)
    contour!(x1, x2, (x1, x2) -> model(W, [x1, x2]')[1], label = "", colorbar_ticks=([1], [0.0]))
end
