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
    x1 = [1,2,6,8,10] #Note that each observation is now a real number
    x2 = [1,2,6,8,5]
    theta = (1, 0.000001, [1])
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
    theta = (1, 2, [1])

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
    theta = (1, 2, [1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)
    @test isapprox(log_m_likelihood(gaussian_process), -14.828870588181731)
    
end

@testset "grad_rbf implemenation" begin
    x = [1,2,6,8,10]
    y = sin.(x) + x
    theta = (1, 2, [1])

    gp = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)

    @test isapprox(grad_rbf(x[1,:], gp.X[2, :], gp.theta), [0.0, 0.0, -0.36787944117144233])
    @test isapprox(grad_rbf(x[5,:], gp.X[4, :], gp.theta), [0.0, 0.0, -0.07326255555493671])
    @test isapprox(grad_rbf(x[3,:], gp.X[3, :], gp.theta), [0,0,0])
end

@testset "grad_logmlik implementation" begin
    x = [1,2,6,8,10]
    y = sin.(x) + x
    theta = (1, 2, [1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)

    @test isapprox(grad_logmlik(gaussian_process), [-0.0, -0.0, 0.4696765605323389])
end

@testset "predict_gp implementation" begin
    x = [1,2,6,8,10]
    x_pred = [3,5,7,9]
    y = sin.(x) + x
    theta = (1, 2, [1])

    gaussian_process = train_gp(x, vec(y).-mean(y), rbf_kernel, theta)

    pred_mean, variance = predict_gp(gaussian_process,x_pred)
    @test isapprox(pred_mean,[-0.3198864262027336, -0.010048140862259584, 0.38054063837135477, 0.8384452495525727])
    @test isapprox(variance,[2.9546452357668618, 2.954886731886043, 2.910322486486357, 2.910322486486357])
end


# The next testset is in case you want to implement gradient descent
# with status return value. However, if you want to debug your gradient 
# descent in another way there is another test set further down
@testset "gradientdescent w. status" begin
    # Let us minimize the function x[1]^2 + x[2]^2 using gradient descent
    # the gradient ist given by the following function
    f(x) = x[1]^2 + x[2]^2
    f_grad(x) = [2*x[1], 2*x[2]]

    ## First Check: moving in right direction
    x_star, status = gradientdescent(f_grad, 
    [1, 20],
    [[0, 0] [100, 100]],
    stepsize = 1e-10,
    eps = 1e-1,
    max_iter = 2)

    # After two tiny steps value of the function should be smaller than at initialization:
    @test f(x_star) < f([1, 20])

    # After some steps, x_star should be "relatively" close to 0 ;)
    x_star, status = gradientdescent(f_grad, 
                    [1, 20],
                    [[0, 0] [100, 100]],
                    stepsize = 1e-1,
                    eps = 1e-1,
                    max_iter = 50)


    @test norm(x_star) < 1e-1
    @test status == 1

    # be careful about stepsize
    x_star, status = gradientdescent(f_grad, 
                    [1, 20],
                    [[0, 0] [100, 100]],
                    stepsize = 1e1,
                    eps = 1e-1,
                    max_iter = 2)

    @test status == 10


    # be careful about stepsize
    x_star, status = gradientdescent(f_grad, 
                    [-1, -20],
                    [[0, 0] [100, 100]],
                    stepsize = 1e1,
                    eps = 1e-1,
                    max_iter = 2)



    @test status == -10
end


# Gradient descent without status, uncomment if applicable.
# @testset "gradientdescent w/0 status" begin
#     # Let us minimize the function x[1]^2 + x[2]^2 using gradient descent
#     # the gradient ist given by the following function
#     f(x) = x[1]^2 + x[2]^2
#     f_grad(x) = [2*x[1], 2*x[2]]

#     ## First Check: moving in right direction
#     x_star = gradientdescent(f_grad, 
#     [1, 20],
#     [[0, 0] [100, 100]],
#     stepsize = 1e-10,
#     eps = 1e-1,
#     max_iter = 2)

#     # After two tiny steps value of the function should be smaller than at initialization:
#     @test f(x_star) < f([1, 20])

#     # After some steps, x_star should be "relatively" close to 0 ;)
#     x_star = gradientdescent(f_grad, 
#                     [1, 20],
#                     [[0, 0] [100, 100]],
#                     stepsize = 1e-1,
#                     eps = 1e-1,
#                     max_iter = 50)


#     @test norm(x_star) < 1e-1

# end
# @testset "find_theta implementation" begin
#     x = [1,2,6,8,10]
#     y = sin.(x) + x
#     theta = (1, 2, [1])
#     result = find_theta(x,y,theta)
    
#     @test isapprox(result[1], [1.0, 2.0, 0.33575944838962485])
# end