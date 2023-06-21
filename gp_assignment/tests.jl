include("../src/GP.jl")
using Distributions

using Test

@testset "RBF Kernel implementation" begin
    x1 = [1,2,6,8,10]
    x2 = [1,2,6,8,5]
    theta = (1, 2, [1, 1, 1, 1, 1])
    result = rbf_kernel(x1,x2, theta)
    @test isapprox(result, 1.3887943864964021e-11)
end

@testset "OU Kernel implementation" begin
    x1 = [1,2,6,8,10]
    x2 = [1,2,6,8,5]
    theta = (1, 2, [1, 1, 1, 1, 1])
    result = ou_kernel(x1,x2, theta)
    @test isapprox(result, 0.006737946999085467)
end

@testset "kernelmat implementation" begin
    x1 = [1,2,6,8,10]
    x2 = [1,2,6,8,5]
    theta = (1, 0.000001, [1, 1, 1, 1, 1])
    C = kernelmat(x1, x2, rbf_kernel, theta)
    @test size(C) == (5,5)
    @test isapprox(1.000001, C[1,1])
    @test isapprox(0.36787944117144233, C[1,2])
    @test isapprox(0.36787944117144233, C[2,1])
    @test isapprox(1.3887943864964021e-11, C[1,3])
    @test isapprox(1.3887943864964021e-11, C[3,1])
end

@testset "train_gp implementation" begin
    x = [1,2,6,8,10]
    y = sin.(x) + x
    theta = (1, 2, [1, 1, 1, 1, 1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)
    @test isapprox(gaussian_process.X, x)
    @test isapprox(gaussian_process.y, vec(y).-mean(y))
    @test gaussian_process.kernel == rbf_kernel
    @test gaussian_process.theta ==  theta
    @test isapprox(gaussian_process.L[1:4], [1.7320508075688772, 0.21239529438966134, 8.01820812892739e-12, 3.026981449073326e-22])
end

@testset "log_m_likelihood implemenation" begin
    x = [1,2,6,8,10]
    y = sin.(x) + x
    theta = (1, 2, [1, 1, 1, 1, 1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)
    @test isapprox(log_m_likelihood(gaussian_process), -14.828870588181731)
    
end

@testset "grad_rbf implemenation" begin
    x = [1,2,6,8,10]
    y = sin.(x) + x
    theta = (1, 2, [1, 1, 1, 1, 1])

    gp = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)

    @test isapprox(grad_rbf(x[1,:], gp.X[2, :], gp.theta), [0.0, 0.0, -0.36787944117144233])
    @test isapprox(grad_rbf(x[5,:], gp.X[4, :], gp.theta), [0.0, 0.0, -0.07326255555493671])
    @test isapprox(grad_rbf(x[3,:], gp.X[3, :], gp.theta), [0,0,0])
end

@testset "grad_loglik implementation" begin
    x = [1,2,6,8,10]
    y = sin.(x) + x
    theta = (1, 2, [1, 1, 1, 1, 1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)

    @test isapprox(grad_loglik(gaussian_process), [-0.0, -0.0, 0.4696765605323389])
end

@testset "predict_gp implementation" begin
    x = [1,2,6,8,10]
    x_pred = [3,5,7,9]
    y = sin.(x) + x
    theta = (1, 2, [1, 1, 1, 1, 1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)

    pred_mean, variance = predict_gp(gaussian_process,x_pred)
    @test isapprox(pred_mean,[-0.3198864262027336, -0.010048140862259584, 0.38054063837135477, 0.8384452495525727])
    @test isapprox(variance,[2.9546452357668618, 2.954886731886043, 2.910322486486357, 2.910322486486357])
end

# @testset "find_theta implementation" begin
#     x = [1,2,6,8,10]
#     y = sin.(x) + x
#     theta = (1, 2, [1])
#     result = find_theta(x,y,theta)
    
#     @test isapprox(result[1], [1.0, 2.0, 0.33575944838962485])
# end