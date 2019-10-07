using TensorToolbox
using LinearAlgebra

include("orthogonal_complement.jl")
include("hooi.jl")

function cont(A,B,m,res) #縮約するところm,残る次元 res(行ベクトル)
    amatrix = tenmat(A,row=m)'
    bmatrix = tenmat(B,row=m)
    cmatrix = amatrix*bmatrix
    if length(size(cmatrix)) == 1
        return cmatrix
    else
        if length(res) == 1 #行列が返ってくる
            sz = [size(A,res[1]) size(B,res[1])]
            order = [1 2] # <-
            tensor = reshape(cmatrix,sz[1],sz[2])
        
        elseif length(res) == 2 #４階テンソルが返ってくる
            sz = [size(A,res[1]) size(A,res[2]) size(B,res[1]) size(B,res[2])]
            order = [1 2 3 4] # <-
            tensor = reshape(cmatrix,sz[order[1]],sz[order[2]],sz[order[3]],sz[order[4]]) # <-
        end
    end
    
    return tensor
end

function objectF(A,X,Y,Z)
    A = tucker(A,X',Y',Z')
    f = -norm(A)^2/2
    return f
end


function movePoint2(T1,T2,s,v,t)
    M = (T1*diagm(0 => cos.(t*s)) + T2*diagm(0 => sin.(t*s)))*v'
    return M
end

function moveBase(B,T1,T2,U,s,t)
    B = ((T1*diagm(0 => -sin.(t*s))) + T2*diagm(0 => cos.(t*s)))*U' + B - T2*U'
    return B
end

function orthonormalize(X)
    if abs(norm(X'*X - Matrix{Float64}(I,size(X,2),size(X,2)))) > 1e-12 #Xが直交行列でなくなっているとき
        U,s,V = svd(X)
        X = U
    end
    return X
end

function gradhess2(A,X,Y,Z,Xvert,Yvert,Zvert)
    #仮
    Bz = ttm(A,X',1)
    By = ttm(Bz,Y',2)
    Cyz = ttm(Bz,Yvert',2)
    #完成
    F = ttm(By,Z',3)
    Bz = ttm(By,Zvert',3)
    By = ttm(Cyz,Z',3)
    Cyz = ttm(Cyz,Zvert',3)
    #仮
    Bx = ttm(A,Xvert',1)
    Cxz = ttm(Bx,Y',2)
    Cxy = ttm(Bx,Yvert',2)
    #完成
    Bx = ttm(Cxz,Z',3)
    Cxz = ttm(Cxz,Zvert',3)
    Cxy = ttm(Cxy,Z',3)
    
    #gradient
    grx = -cont(Bx,F,[2,3],[1])
    gry = -cont(By,F,[1,3],[2])
    grz = -cont(Bz,F,[1,2],[3])
    
    #Hessian
    Hxx = kron(Matrix{Float64}(I,size(X,2),size(X,2)),cont(Bx,Bx,[2,3],1)) - kron(cont(F,F,[2,3],1),Matrix{Float64}(I,size(X,1)-size(X,2),size(X,1)-size(X,2)))
    Hyy = kron(Matrix{Float64}(I,size(Y,2),size(Y,2)),cont(By,By,[1,3],2)) - kron(cont(F,F,[1,3],2),Matrix{Float64}(I,size(Y,1)-size(Y,2),size(Y,1)-size(Y,2)))
    Hzz = kron(Matrix{Float64}(I,size(Z,2),size(Z,2)),cont(Bz,Bz,[1,2],3)) - kron(cont(F,F,[1,2],3),Matrix{Float64}(I,size(Z,1)-size(Z,2),size(Z,1)-size(Z,2)))
    
    Hxy = tenmat(cont(Cxy,F,3,[1,2]),row=[1,3],col=[2,4]) + tenmat(cont(Bx,By,3,[1,2]),row=[1,3],col=[4,2])
    Hxz = tenmat(cont(Cxz,F,2,[1,3]),row=[1,3],col=[2,4]) + tenmat(cont(Bx,Bz,2,[1,3]),row=[1,3],col=[4,2])
    Hyz = tenmat(cont(Cyz,F,1,[2,3]),row=[1,3],col=[2,4]) + tenmat(cont(By,Bz,1,[2,3]),row=[1,3],col=[4,2])
    
    H1 = hcat(Hxx,Hxy,Hxz)
    H2 = hcat(Hxy',Hyy,Hyz)
    H3 = hcat(Hxz',Hyz',Hzz)
    H = -vcat(H1,H2,H3);
    
    return grx,gry,grz,H,Xvert,Yvert,Zvert
end

function ngsolver(grx,gry,grz,H)
    #準備
    xn = size(grx,1)
    xp = size(grx,2)
    yn = size(gry,1)
    yp = size(gry,2)
    zn = size(grz,1)
    zp = size(grz,2)
    
    #変形
    g1 = reshape(grx,xn*xp,1)
    g2 = reshape(gry,yn*yp,1)
    g3 = reshape(grz,zn*zp,1)
    
    g = -[g1;g2;g3]
    
    #ムーア・ペンローズの疑似逆行列
    U,S,V = svd(H,full=true)
    r = rank(diagm(0 => S))
    x = V[:,1:r]*((U[:,1:r]'*g)./(S[1:r]))
    
    Dx = reshape(x[1 : xn*xp], xn, xp)
    Dy = reshape(x[xn*xp+1 : xn*xp+yn*yp], yn, yp)
    Dz = reshape(x[xn*xp+yn*yp+1 : end], zn, zp)
    
    return Dx,Dy,Dz
end

function ngtensor(A,X,Y,Z,r,t,method)
    #準備
    xn = size(X,1)
    xp = size(X,2)
    yn = size(Y,1)
    yp = size(Y,2)
    zn = size(Z,1)
    zp = size(Z,2)
    rel = zeros(r+1,1)
    
    Xv = orthogonal_complement(X)[:,size(X,2)+1:size(X,1)]
    Yv = orthogonal_complement(Y)[:,size(Y,2)+1:size(Y,1)]
    Zv = orthogonal_complement(Z)[:,size(Z,2)+1:size(Z,1)]
    
    for its = 1:r
        
        #println(its)

        grx,gry,grz,H = gradhess2(A,X,Y,Z,Xv,Yv,Zv)
        
        Dx,Dy,Dz = ngsolver(grx,gry,grz,H)
        
        if its == 1
            rel[1] = (norm(grx)+norm(gry)+norm(grz))/-objectF(A,X,Y,Z)
        end
        
        if method == 1
            qx,r = qr(X+Xv*Dx)
            X = qx[:,1:xp]
            
            qy,r = qr(Y+Yv*Dy)
            Y = qy[:,1:yp]
            
            qz,r = qr(Z+Zv*Dz)
            Z = qz[:,1:zp]
            
        elseif method == 2
            ux,sx,vx = svd(Dx)
            uy,sy,vy = svd(Dy)
            uz,sz,vz = svd(Dz)
            
            #Xに関して
            T1x = X*vx
            T2x = Xv*ux
            
            X = movePoint2(T1x,T2x,sx,vx,t)
            Xv = moveBase(Xv,T1x,T2x,ux,sx,t)
            X = orthonormalize(X)

            #Yに関して
            T1y = Y*vy
            T2y = Yv*uy
            
            Y = movePoint2(T1y,T2y,sy,vy,t)
            Yv = moveBase(Yv,T1y,T2y,uy,sy,t)
            Y = orthonormalize(Y)
            
            #Zに関して
            T1z = Z*vz
            T2z = Zv*uz
            
            Z = movePoint2(T1z,T2z,sz,vz,t)
            Zv = moveBase(Zv,T1z,T2z,uz,sz,t)
            Z = orthonormalize(Z)

        end
        
        rel[its+1] = (norm(grx)+norm(gry)+norm(grz))/-objectF(A,X,Y,Z)
        #println(norm(grx)+norm(gry)+norm(grz))
        
    end
    
    return X,Y,Z,rel
end