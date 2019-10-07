using LinearAlgebra

function project_along(b,v)
    #bのvに沿った射影を返す
    ip = v'*v
    if ip>1e-15
        sigma = (b'*v)/ip
    else
        sigma = 0
    end
    return sigma*v
end

function project_orthogonal(b,vlist)
    #与えたベクトルのリストvlistに直交する射影を返す
    for i = 1:size(vlist,2)
        b = b - project_along(b,vlist[:,i])
        b = b/norm(b)
    end
    return b
end

function orthogonalize(vlist)
    vstarlist = vlist[:,1]
    for i = 2:size(vlist,2)
        v = vlist[:,i]
        tmp = project_orthogonal(v,vstarlist)
        vstarlist =  hcat(vstarlist,tmp)
    end
    return vstarlist
end

function orthogonal_complement(U_basis)
    #WにおけるUの直交補空間の基底を並べた行列Vを返す関数
    #Ubasisが直交行列，Wがn次元空間の基底を並べた行列
    W_basis =  Matrix{Int8}(I, size(U_basis,1), size(U_basis,1)-size(U_basis,2))
    vlist = hcat(U_basis,W_basis)
    return orthogonalize(vlist)
end
