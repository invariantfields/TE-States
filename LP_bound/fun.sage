def comb(n, k): # binomial(n,k)
    if n >= k and k >= 0 and n >= 0:
        value = binomial(n, k)
    else:
        value = 0
    return value
    

def K2(m, k, n): # Krawtchouk polynomials
    ref = 0
    for alp in range(0,n+1):
        ref += (-1)**alp*comb(n-k, m-alp)*comb(k, alp)
        
    return ref

def Standard_form(n,k): # constraints in the standard form A*a' >= b where a' are the variables. c*a' is the objective function. 
    
    A = matrix(QQ, 2, n+1)
    
    # a'[0] = 1
    
    A[0, 0] = QQ(1)
    A[1, 0] = QQ(-1)
    
    b = matrix(QQ, 2, 1)
    b[0, 0] = QQ(1)
    b[1, 0] = QQ(-1)
    
    # a'[i] = a'[n-i]
    
    for i in range(0, ceil(n/2)):
        z = matrix(QQ, 1, n+1)
        z[0, n-i] = QQ(1)
        z[0, i] = QQ(-1)
    
        A = A.stack(z)
        b = b.stack(matrix(QQ, [[0]]))
    
        z = matrix(QQ, 1, n+1)
        z[0, n-i] = QQ(-1)
        z[0, i] = QQ(1)
    
        A = A.stack(z)
        b = b.stack(matrix(QQ, [[0]]))
    
     # a[i] >=0
    for j in range(1, n+1):
        z = vector(QQ, [
            (-1)^ (j-r) * 2^r * comb(n-r, n-j)
            for r in range(0, n+1)
        ])
    
        A = A.stack(z)
        b = b.stack(matrix(QQ, [[0]]))
    
    # s[i] >=0
    for j in range(0, n+1):
        z = vector(QQ, [
            K2(n-j, k, n)
            for k in range(0, n+1)
        ])
    
        A = A.stack(z)
        b = b.stack(matrix(QQ, [[0]]))

    # s[n-j] = 0 for j odd

    odd = [
        i for i in range(0, n+1)
        if ((n-i) % 2) == 1
    ]
    
    
    for j in odd:
        z = vector(QQ, [
            -K2(n-j, k, n)
            for k in range(0, n+1)
        ])
    
        A = A.stack(z)
        b = b.stack(matrix(QQ, [[0]]))

    # objective function, i.e. c*a' = a[floor(n/2)]
    c = matrix(QQ, 1, n+1)
    c[0, k] = QQ(1)

    return A, b, c