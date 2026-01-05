# Step 3: Build coefficient matrix
    if verbose:
        print("  Building coefficient matrix...")
    
    mono_to_idx = {mono: i for i, mono in enumerate(monomial_list)}
    v = []
    r = []
    c = []
    for col, expansion in enumerate(expansions):
        for mono, coeff in expansion.terms.items():
            if coeff != 0:
                v.append(coeff)
                r.append(mono_to_idx[mono])
                c.append(col)
            
    coo = coo_matrix((v,(r,c)),shape =(len(monomial_list),len(expansions)))
    
    # Step 4: Compute null space
    if verbose:
        print("  Computing null space via SVD for sparse...")
    
    U, s, Vt = svds(coo, k = min(coo.shape)-1, which="SM")
    tol = 1e-12
    null_space = s < tol
    null_vec =  Vt.T[:, null_space]

    
    if verbose:
        print(f"  Matrix shape: {coo.shape}")
        print(f"  Null space dimension: {null_space.shape[1]}")
    
    prod = coo @ null_vec

    if not np.allclose(prod, np.zeros(prod.shape), tol):
        print(prod)
        print("ERROR: SVD did not compute nullspace correctly!")
        exit(1)


#reconstruct now the intersection
    # Step 5: Reconstruct intersection elements
    intersection_basis = []
    n1 = len(basis1)
    
    for col_idx in range(null_space.shape[1]):
        null_vector = null_space[:, col_idx]
        int_coeffs = rationalize_vector(null_vector)
        
        # Build linear combination from basis1 side
        result = Polynomial.zero()
        for i in range(n1):
            if int_coeffs[i] != 0:
                result = result + (int_coeffs[i] * (basis1[i] * g1))
        
        if not result.is_zero():
            intersection_basis.append(result.expand())
    
    if verbose:
        print(f"  Intersection dimension: {len(intersection_basis)}")
    
    return intersection_basis