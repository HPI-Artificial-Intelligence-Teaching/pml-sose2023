# Cholesky decomposition
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

"""
    L = cholesky_crout(A)

Computes the Cholesky-Crout decomposition of a symmetric, positive definite matrix `A`

# Examples
```jldoctest
julia> cholesky_crout(hcat([1, 2, 0],[2,20,4],[0,4,17]))
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 2.0  4.0  0.0
 0.0  1.0  4.0
```
"""
function cholesky_crout(A::Matrix)
    # check that the matrix is square
    if (size(A)[1] != size(A)[2])
        error("matrix must be square")
    end
    n = size(A)[1]

    # create a zero matrix
    L = zeros(n, n)

    # run the Cholesky decomposition
    for j = 1:n
        sum = 0
        for k = 1:(j-1)
            sum += L[j,k] * L[j,k]
        end
        L[j,j] = sqrt(A[j,j] - sum)

        for i = (j+1):n
            sum = 0
            for k = 1:(j-1)
                sum += L[i,k] * L[j,k]
            end
            L[i,j] = (A[i,j] - sum) / L[j,j]
        end
    end

    return (L)
end

"""
    L = cholesky_crout(A)

Computes the Cholesky-Banachiewicz decomposition of a symmetric, positive definite matrix `A`

# Examples
```jldoctest
julia> cholesky_banachiewicz(hcat([1, 2, 0],[2,20,4],[0,4,17]))
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 2.0  4.0  0.0
 0.0  1.0  4.0
```
"""
function cholesky_banachiewicz(A::Matrix)
    # check that the matrix is square
    if (size(A)[1] != size(A)[2])
        error("matrix must be square")
    end
    n = size(A)[1]

    # create a zero matrix
    L = zeros(n, n)

    # run the Cholesky decomposition
    for i = 1:n
        for j = 1:i
            sum = 0
            for k = 1:(j-1)
                sum += L[i,k] * L[j,k]
            end
        
            L[i,j] = (i == j) ? sqrt(A[j,j] - sum) : (A[i,j] - sum)/L[j,j]
        end
    end

    return (L)
end
