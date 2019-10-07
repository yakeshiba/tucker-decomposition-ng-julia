using LinearAlgebra
using TensorToolbox

function tucker(A,X,Y,Z)
    B = ttm(A,X,1)
    B = ttm(B,Y,2)
    B = ttm(B,Z,3)
    return B
end

function trhosvd(A,p,q,r)
    #モード1行列化
    A1 = tenmat(A,1)
    U1,S1,V1 = svd(A1)
    U1 = U1[:,1:p]

    #モード2行列化
    A2 = tenmat(A,2)
    U2,S2,V2 = svd(A2)
    U2 = U2[:,1:q]

    #モード3行列化
    A3 = tenmat(A,3)
    U3,S3,V3 = svd(A3)
    U3 = U3[:,1:r]
    
    #コアテンソル
    G = tucker(A,U1',U2',U3')

    return G,U1,U2,U3
end

function hooi(A,i,j,k,p)

    #初期値
    G,A1,A2,A3 = trhosvd(A,i,j,k)

    for its = 1:p
        #一つ目
        tmp1 = ttm(A,A2',2)
        Y1 = ttm(tmp1,A3',3)
        U1,S1,V1 = svd(tenmat(Y1,1))
        A1 = U1[:,1:i]

        #二つ目
        tmp2 = ttm(A,A1',1)
        Y2 = ttm(tmp2,A3',3)
        U2,S2,V2 = svd(tenmat(Y2,2))
        A2 = U2[:,1:j]

        #三つ目
        tmp3 = ttm(A,A1',1)
        Y3 = ttm(tmp3,A2',2)
        U3,S3,V3 = svd(tenmat(Y3,3))
        A3 = U3[:,1:k]
    end

    G = tucker(A,A1',A2',A3')

    return G,A1,A2,A3
end

