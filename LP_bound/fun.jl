# Mapping Functions

function f4_fun(n)
    funct = Dict()
    size = 0
    for i in 0:n
        for j in 0:i
            for t in (i + j - n):min(i, j)
                for p in 0:t
                    funct[(i, j, t, p)] = size
                    size += 1
                end
            end
        end
    end
    return funct, size
end

function f2_fun(n)
    fun = Dict{Tuple{Int, Int}, Int}()
    count = 0
    for a in 0:n
        for k in a:div(n + a, 2)
            fun[(a, k)] = count
            count += 1
        end
    end
    return fun, count
end

function multi(dic,cte)
    S=Dict()
    
    for k in keys(dic)
        S[k]=cte*dic[k]
    end 
    
    return S 
end 

# Combinatorial Functions

function comb(n, k)
    
    if n>=k && k>=0 && n>=0
        value=binomial(n,k)
    else
        value=0
    end
    
    return value
end

function multinomial(n,i,j,k,l)

    value=factorial(big(n))/(factorial(big(i))*factorial(big(j))*factorial(big(k))*factorial(big(l))*factorial(big(n-i-j-k-l)))
    
    return big(value)
end

function gamma(i, j, t, p, n)
    
    value= (big(3)//big(1))^(i + j - t) * (big(2)//big(1))^(t - p) * multinomial(n, p, t - p, i - t, j - t)
    return big(value)
    
end


function beta(i, j, k, m, t)
    b = 0
    for u in 0:m
        b += comb(u, t) * comb(m - 2*k, u - k) * comb(m - k - u, i - u) * comb(m - k - u, j - u)*(-1)^(t - u)
    end
    return b
end

function alpha(i, j, t, p, a, k, n)
    
    sum = 0
    
    for g in 0:p
        sum += (-1//1)^(a - g)*comb(a, g)*comb(t - a, p - g)*(2//1)^(t - a - p + g)
    end
    
    FF=sqrt(big(3))
    value=beta(i - a, j - a, k - a, n - a, t - a)*(FF^(i+j)/3^t)*sum
    
    return big(value)
 
end

function Kw(m, k, n)
    sum = 0
    for alp in 0:n
        sum += (-1//1)^alp*comb(n-k, m-alp)*comb(k, alp)*(3//1)^(m-alp)
    end
    return big(sum)
end

function K2(m, k, n)
    sum = 0
    for alp in 0:n
        sum += (-1)^alp*comb(n-k, m-alp)*comb(k, alp)
    end
    return (sum)
end

function kron_delta(a, b)
    if a == b
        return 1
    else
        return 0
    end
end

function alphamatrices(x, f4, n)

    Mat=[]

    for a in 0:n
        for k in a:div((n+a),2)
    
            matrix=Matrix(undef,n + a - 2*k+1,n + a - 2*k+1)

            for i in k:n+a-k
                for j in k:i
    
                    suma = 0
    
                    for t in (i+j-n):min(i,j)
                        for p in 0:t
                           
                            suma += alpha(i,j,t,p,a,k,n)*x[f4[i, j, t, p]]
                        end
                    end
                   
             
                    matrix[i-k+1,j-k+1] = suma
                    matrix[j-k+1,i-k+1] = suma
                    
                end
            end

            push!(Mat, matrix)
        end
    end
    
    return Mat
end
